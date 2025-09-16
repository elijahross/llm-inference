cat > test_handler.py <<'EOF'
import os
import logging

# Set Hugging Face cache to workspace/
os.environ["HUGGINGFACE_HUB_CACHE"] = "workspace/hf_cache"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from pathlib import Path

logging.info("Initializing Runner...")

runner = Runner(
    which=Which.Plain(
        model_id="/workspace/hf_cache/models--EricB--Llama-3.2-1B-Instruct-UQFF/snapshots/fc33d73fbe7445f9cda60755d52a8203f5e87778",
        arch=Architecture.Llama,
        from_uqff="llama4-scout-instruct-q4k-0.uqff; llama4-scout-instruct-q4k-1.uqff; llama4-scout-instruct-q4k-2.uqff; llama4-scout-instruct-q4k-3.uqff; llama4-scout-instruct-q4k-4.uqff; llama4-scout-instruct-q4k-5.uqff; llama4-scout-instruct-q4k-6.uqff"
    ),
)

logging.info("Runner initialized successfully.")

def handler(job=None):
    try:
        logging.info("Building job input...")
        job_input = {
            "model": "default",
            "messages": [
                {"role": "user", "content": "Tell me a story about the Rust type system."}
            ],
            "max_tokens": 256,
            "presence_penalty": 1,
            "top_p": 0.1,
            "temperature": 0.1,
        }
        logging.debug(f"Job input: {job_input}")

        logging.info("Creating ChatCompletionRequest...")
        req = ChatCompletionRequest(
            model=job_input["model"],
            messages=job_input["messages"],
            max_tokens=job_input.get("max_tokens", 12096),
            presence_penalty=job_input.get("presence_penalty", 0.0),
            top_p=job_input.get("top_p", 1.0),
            temperature=job_input.get("temperature", 1.0),
        )
        logging.debug(f"Request object: {req}")

        logging.info("Sending request to Runner...")
        res = runner.send_chat_completion_request(req)

        output = res.choices[0].message.content
        logging.info("Inference completed successfully.")
        logging.debug(f"Model response: {output}")

        return {"output": output}

    except Exception as e:
        logging.error("Error during handler execution", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logging.info("Starting local test of handler...")
    result = handler()
    logging.info("Handler finished.")
    print("Final result:", result)
EOF