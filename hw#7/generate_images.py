
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

MODEL_ID = os.getenv("SD_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers")
OUTPUT_DIR = Path(__file__).parent / "images"
SEED = 42

NEGATIVE_PROMPT = (
    "blurry, low quality, lowres, jpeg artifacts, deformed, bad anatomy, "
    "extra fingers, watermark, text, signature, oversaturated"
)


EXPERIMENTS = [
    {
        "name": "1_cat",
        "basic": "a cat",
        "improved": (
            "portrait of a fluffy ginger cat sitting on a sunny windowsill, "
            "soft morning light, shallow depth of field, 85mm lens, "
            "highly detailed fur, professional pet photography"
        ),
    },
    {
        "name": "2_lisbon",
        "basic": "Lisbon street with a tram",
        "improved": (
            "yellow vintage tram 28 climbing a narrow cobblestone street in Lisbon, "
            "colorful azulejo-tiled houses, golden hour, warm sunlight, "
            "watercolor illustration, loose brush strokes, pastel palette"
        ),
    },
    {
        "name": "3_robot_dev",
        "basic": "robot programmer",
        "improved": (
            "a small friendly robot programmer working late at night at a wooden desk, "
            "two monitors showing Python code, a cup of coffee and a rubber duck on the desk, "
            "cozy room lit by a desk lamp and neon city lights outside the window, "
            "cinematic lighting, pixar style 3d render, wide shot, rule of thirds"
        ),
    },
]


def get_client(token: str | None = None) -> InferenceClient:
    load_dotenv()
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Нет HF_TOKEN: создай токен на huggingface.co/settings/tokens и добавь в .env")
    return InferenceClient(provider="hf-inference", api_key=token)


def generate(client: InferenceClient, prompt: str, negative_prompt: str | None = None):
    return client.text_to_image(
        prompt,
        model=MODEL_ID,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        seed=SEED,
    )


def run_experiments(client: InferenceClient, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for exp in EXPERIMENTS:
        variants = [
            ("basic", exp["basic"], None),
            ("improved", exp["improved"], NEGATIVE_PROMPT),
        ]
        for label, prompt, negative in variants:
            path = output_dir / f"{exp['name']}_{label}.png"
            print(f"[{path.name}] {prompt}")
            try:
                generate(client, prompt, negative).save(path)
                saved.append(path)
            except Exception as error:
                print(f"  ошибка: {error}")
    return saved


def main() -> None:
    saved = run_experiments(get_client())
    print(f"\nГотово: {len(saved)} изображений в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
