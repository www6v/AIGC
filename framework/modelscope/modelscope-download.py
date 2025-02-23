# 从modelscope上下载模型
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import MsDataset

###### model

# model_dir = snapshot_download('baichuan-inc/baichuan-7B', cache_dir='./model', revision='master')
# model_dir = snapshot_download('ZhipuAI/chatglm2-6b', cache_dir='./model', revision='master')
# model_dir = snapshot_download('xrunda/m3e-base', cache_dir='./model', revision='master')


model_dir = snapshot_download('ZhipuAI/chatglm2-6b-int4', cache_dir='./model', revision='master')
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='./model', revision='master')


model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', cache_dir='./model', revision='master')
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='./model', revision='master')


##### dataset 加载数据集
dataset_dir = snapshot_download('', cache_dir='./dataset', revision='master')


#  modelscope download --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT'  --local_dir './local_dir'

ds = MsDataset.load('afqmc_small', split='train')