import os
import json
import time
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
#import ipdb

os.environ["RITS_API_KEY"] = "..."

def get_client(base_url):

    client = OpenAI(
        api_key=os.environ["RITS_API_KEY"],
        default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
        base_url=base_url,
    )

    return client


def generate_rits_response(client, model_name, prompt, max_tokens=1000, batch=False):
    try:
        #start_time = time.time()
        # ipdb.set_trace()
        completion = client.completions.create(
            # this must match the custom deployment name you chose for your model
            model=model_name,
            prompt=prompt,
            #temperature=temperature,
            max_tokens=max_tokens
        )
        end_time = time.time()

        #execution_time = end_time - start_time
        #print(f"function execution time from RITS: {execution_time} seconds")

        # ipdb.set_trace()
        resp = completion.model_dump_json()
        resp = json.loads(resp)

        if not batch: return resp['choices'][0]['text']
        else:  # batch
            return [r['text'] for r in resp['choices']]

    except Exception as exception:
        return dict(error=exception)


# sasvpn-fast.emea.ibm.com/TUNNELALL
#base_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3-2/v1'
#base_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b-a100/v1'
#base_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01/v1'
#base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-5-0-120b-sft/v1'
base_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1'
#base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/google-gemma-4-31b-a100/v1'

#model_name = 'deepseek-ai/DeepSeek-V3.2'
#model_name = 'openai/gpt-oss-120b-a100'
#model_name = 'mistralai/mixtral-8x22B-instruct-v0.1'
#model_name = 'ibm-research/granite-5.0-120b-sft'
model_name = 'meta-llama/llama-3-3-70b-instruct'
#model_name = 'google/gemma-4-31B'


if __name__ == '__main__':
    client = get_client(base_url)
    print(f'inference endpoint: {client.base_url}')  # inference endpoint
    #prompt = "what model are you -- give me name and size"
    prompt = "do you have agentic capabilities? tool calling, basic reasoning, etc."
    response = generate_rits_response(client, model_name, prompt, max_tokens=500)
    print(f'response: {response}')
