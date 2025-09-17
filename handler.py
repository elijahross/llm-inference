import runpod
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture
import os

# You can initialize the runner once, so it’s cached in memory if the pod stays warm.
runner = Runner(
    which=Which.VisionPlain(
        model_id="EricB/Llama-4-Scout-17B-16E-Instruct-UQFF",
        from_uqff=["/runpod-volume/models/llama4-scout-instruct-q4k-0.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-1.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-2.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-3.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-4.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-5.uqff", "/runpod-volume/models/llama4-scout-instruct-q4k-6.uqff"],
        arch=VisionArchitecture.Llama4,
        hf_cache_path="/runpod-volume/hf_cache"
    )
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
            "output": res,
            "response": res.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}

# Required by RunPod
runpod.serverless.start({"handler": handler})
