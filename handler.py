import runpod
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

# You can initialize the runner once, so it’s cached in memory if the pod stays warm.
runner = Runner(
    which=Which.VisionPlain(
         model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        arch=VisionArchitecture.Llama4,
        quantized_filename="llama4-scout-instruct-q4k-0.uqff;llama4-scout-instruct-q4k-1.uqff;llama4-scout-instruct-q4k-2.uqff;llama4-scout-instruct-q4k-3.uqff;llama4-scout-instruct-q4k-4.uqff;llama4-scout-instruct-q4k-5.uqff;llama4-scout-instruct-q4k-6.uqff",
    ),
)

def handler(job):
    job_input = job["input"]

    # Build ChatCompletionRequest
    req = ChatCompletionRequest(
        model="default",
        messages=job_input["messages"],  # Expect same structure as OpenAI-style API
        max_tokens=job_input.get("max_tokens", 120256),
        presence_penalty=job_input.get("presence_penalty", 0.0),
        top_p=job_input.get("top_p", 1.0),
        temperature=job_input.get("temperature", 1.0),
    )

    # Run inference
    res = runner.send_chat_completion_request(req)

    return {
        "output": res.choices[0].message.content,
        "usage": res.usage.__dict__,  # Token usage info
    }

# Required by RunPod
runpod.serverless.start({"handler": handler})
