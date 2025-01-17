import base64
import os
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from aphrodite.common.config import ModelConfig
from aphrodite.multimodal.base import MultiModalDataDict

APHRODITE_IMAGE_FETCH_TIMEOUT = int(
    os.getenv("APHRODITE_IMAGE_FETCH_TIMEOUT", 10))


class ImageFetchAiohttp:
    aiohttp_client: Optional[aiohttp.ClientSession] = None

    @classmethod
    def get_aiohttp_client(cls) -> aiohttp.ClientSession:
        if cls.aiohttp_client is None:
            timeout = aiohttp.ClientTimeout(
                total=APHRODITE_IMAGE_FETCH_TIMEOUT)
            connector = aiohttp.TCPConnector()
            cls.aiohttp_client = aiohttp.ClientSession(timeout=timeout,
                                                       connector=connector)

        return cls.aiohttp_client

    @classmethod
    async def fetch_image(cls, image_url: str) -> Image.Image:
        """Load PIL image from a url or base64 encoded openai GPT4V format"""

        if image_url.startswith('http'):
            parsed_url = urlparse(image_url)
            if parsed_url.scheme not in ["http", "https"]:
                raise ValueError("Invalid 'image_url': A valid 'image_url' "
                                 "must have scheme 'http' or 'https'.")
            # Avoid circular import
            from aphrodite import __version__ as APHRODITE_VERSION

            client = cls.get_aiohttp_client()
            headers = {"User-Agent": f"aphrodite/{APHRODITE_VERSION}"}

            async with client.get(url=image_url, headers=headers) as response:
                response.raise_for_status()
                image_raw = await response.read()
            image = Image.open(BytesIO(image_raw))

        # Only split once and assume the second part is the base64 encoded image
        elif image_url.startswith('data:image'):
            image = load_image_from_base64(image_url.split(',', 1)[1])

        else:
            raise ValueError(
                "Invalid 'image_url': A valid 'image_url' must start "
                "with either 'data:image' or 'http'.")

        image.load()
        return image


def encode_image_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Encode a pillow image to base64 format."""

    buffered = BytesIO()
    if format == 'JPEG':
        image = image.convert('RGB')
    image.save(buffered, format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


# TODO: move this to a model registry for preprocessing vision
# language prompts based on the model type.
def get_full_image_text_prompt(image_prompt: str, text_prompt: str,
                               config: ModelConfig) -> str:
    """Combine image and text prompts for vision language model depending on
    the model architecture."""

    if config.hf_config.model_type in ("llava", "llava_next"):
        full_prompt = f"{image_prompt}\n{text_prompt}"
    elif config.hf_config.model_type == 'phi3_v':
        full_prompt = f"{image_prompt}<s>\n{text_prompt}"
    else:
        raise ValueError(
            f"Unsupported model type: {config.hf_config.model_type}")
    return full_prompt


async def async_get_and_parse_image(image_url: str) -> MultiModalDataDict:
    image = await ImageFetchAiohttp.fetch_image(image_url)
    return {"image": image}
