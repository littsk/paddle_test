import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention 
from paddle.nn.functional.flash_attention import flash_attention, flash_attn_unpadded

import numpy as np

import numba.cuda as cuda

import yaml
import argparse

import os
import traceback


class LoadYaml:
    def __init__(self, yaml_path):
        with open(yaml_path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        try: 
            self.dtype = eval(data['DTYPE'])
        except:
            print(traceback.format_exc())
            raise BaseException("YAML DTYPE is not valid")
        assert isinstance(self.dtype, paddle.dtype), "YAML DTYPE is not valid"
        assert self.dtype == paddle.float16, "DTYPE only support fp16 now"

        self.warmup_steps = data['WARMUP_STEPS']
        self.test_steps = data['TEST_STEPS']
        self.device = data['DEVICE']
        self.dropout_prob = data['DROPOUT_PROB']

        self.training = data['TRAINING']
        self.seed = data['SEED']

class SpeedTest:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', help='config file path')
        parser.add_argument('--test_case', help='which case to test', type=str)
        parser.add_argument('--shape', help="shape of q,k,v")
        opt = parser.parse_args()

        assert os.path.exists(opt.yaml)
        self.cfg = LoadYaml(opt.yaml)

        # need improve
        self.test_case = opt.test_case
        self.shape = tuple(map(int, opt.shape.split(',')))

        self.ctx = cuda.current_context(self.cfg.device)


        def init_input():
            q = paddle.randn(self.shape, dtype=self.cfg.dtype).cuda(self.cfg.device)
            k = paddle.randn(self.shape, dtype=self.cfg.dtype).cuda(self.cfg.device)
            v = paddle.randn(self.shape, dtype=self.cfg.dtype).cuda(self.cfg.device)
            q.stop_gradient = False
            k.stop_gradient = False
            v.stop_gradient = False
            return q, k, v

        def speed_test_fwd_wrapper(func):
            def wrapper(*args, **kwargs):
                # warmup
                free_mem, total_mem = self.ctx.get_memory_info()
                for _ in range(self.cfg.warmup_steps):
                    ret = func(*init_input(), *args, **kwargs)
                self.ctx.synchronize()
                start_event = cuda.event()
                end_event = cuda.event()
                start_event.record()
                for _ in range(self.cfg.test_steps):
                    ret = func(*init_input(), *args, **kwargs)
                end_event.record()
                end_event.synchronize()
                elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)
                used_time_ms = elapsed_time_ms / self.cfg.test_steps
                used_mem_m = (free_mem - self.ctx.get_memory_info()[0]) / 1e6
                return used_time_ms, used_mem_m
            return wrapper

        
        def speed_test_fwd_bwd_wrapper(func):
            def wrapper(*args, **kwargs):
                # warmup
                free_mem, total_mem = self.ctx.get_memory_info()
                for _ in range(self.cfg.warmup_steps):
                    ret = func(*init_input(), *args, **kwargs)
                self.ctx.synchronize()
                start_event = cuda.event()
                end_event = cuda.event()
                start_event.record()
                for _ in range(self.cfg.test_steps):
                    ret = func(*init_input(), *args, **kwargs)
                    if(isinstance(ret, tuple)):
                        ret = ret[0]
                    ret.backward()
                end_event.record()
                end_event.synchronize()
                elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)
                used_time_ms = elapsed_time_ms / self.cfg.test_steps
                used_mem_m = (free_mem - self.ctx.get_memory_info()[0]) / 1e6
                return used_time_ms, used_mem_m
            return wrapper


        
        def attention_naive(q, k, v, dropout_prob=0, scale=None, training=True):
            qt = paddle.transpose(q, [0, 2, 1, 3])
            kt = paddle.transpose(k, [0, 2, 1, 3])
            vt = paddle.transpose(v, [0, 2, 1, 3])
            scale = 1.0 / np.sqrt(q.shape[-1])
            s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
            s = paddle.scale(s, scale)

            dropout_input = F.softmax(s)

            dropout_output = F.dropout(
                x=dropout_input,
                p=dropout_prob,
                training=True,
                mode="upscale_in_train",
            )

            o = paddle.matmul(dropout_output, vt)
            return paddle.transpose(o, [0, 2, 1, 3])

        self.test_cases = dict()
        self.test_cases["core_attn_fwd_test"] = speed_test_fwd_wrapper(attention_naive)
        self.test_cases["core_attn_fwd_bwd_test"] = speed_test_fwd_bwd_wrapper(attention_naive)
        self.test_cases["mem_efficient_attn_fwd_test"] = speed_test_fwd_wrapper(memory_efficient_attention)
        self.test_cases["mem_efficient_attn_fwd_bwd_test"] = speed_test_fwd_bwd_wrapper(memory_efficient_attention)
        self.test_cases["flash_attn_fwd_test"] = speed_test_fwd_wrapper(flash_attention)
        self.test_cases["flash_attn_fwd_bwd_test"] = speed_test_fwd_bwd_wrapper(flash_attention)
    

    def test(self):
        if self.test_case == "core_attn_fwd_test" or self.test_case == "core_attn_fwd_bwd_test":
            used_time_ms, used_mem_m = self.test_cases[self.test_case](dropout_prob=self.cfg.dropout_prob)
        elif self.test_case == "mem_efficient_attn_fwd_test" or self.test_case == "mem_efficient_attn_fwd_bwd_test":
            used_time_ms, used_mem_m = self.test_cases[self.test_case](p=self.cfg.dropout_prob)
        elif self.test_case == "flash_attn_fwd_test" or self.test_case == "flash_attn_fwd_bwd_test":
            used_time_ms, used_mem_m = self.test_cases[self.test_case](dropout=self.cfg.dropout_prob)
        print(self.test_case + ": {0:.2f} ms, {1:.2f} mb".format(used_time_ms, used_mem_m))


if __name__ == "__main__":

    speed_test = SpeedTest()
    speed_test.test()



                
                
                


        # def used layer





