import runpod
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
import os

# You can initialize the runner once, so it’s cached in memory if the pod stays warm.
runner = Runner(
    which=Which.Plain(
        model_id="EricB/Qwen3-32B-UQFF",
        from_uqff=["/runpod-volume/models/qwen332b-q4k-0.uqff", "/runpod-volume/models/qwen332b-q4k-1.uqff"],
        arch=Architecture.Qwen3,
        hf_cache_path="/runpod-volume/hf_cache"
    ),
)

def handler(job):
    try:
        job_input = job["input"]
        hf_token = job.get("token", None)
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        # Build ChatCompletionRequest
        req = ChatCompletionRequest(
            model="default",
            messages=job_input["messages"],
            max_tokens=job_input.get("max_tokens", 12096),  # safer default
            presence_penalty=job_input.get("presence_penalty", 0.0),
            top_p=job_input.get("top_p", 1.0),
            temperature=job_input.get("temperature", 1.0),
        )

        # Run inference
        res = runner.send_chat_completion_request(req)

        return {
            "usage": res.usage,
            "output": res.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}

# Required by RunPod
runpod.serverless.start({"handler": handler})
