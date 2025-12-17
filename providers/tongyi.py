import aiohttp
import json
import asyncio
from typing import Dict, Any, List, Optional
from http import HTTPStatus

from .base import BaseProvider, GenerationConfig, ImageGenerationResult


class TongyiProvider(BaseProvider):
    @property
    def required_config_keys(self) -> list[str]:
        return ["api_key"]
    
    @property
    def default_model(self) -> str:
        return "wanx-2.2"
    
    def validate_config(self) -> bool:
        api_key = self.get_config_value("api_key")
        return isinstance(api_key, str) and api_key.strip() != ""
    
    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        base_url = self.get_config_value("base_url", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis")
        model = config.model or self.get_config_value("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }

        size = self._map_size(config.width, config.height)

        data = {
            "model": model,
            "input": {
                "prompt": config.prompt
            },
            "parameters": {
                "size": size,
                "n": 1,
                "seed": self.get_config_value("seed"),
                "style": config.style
            }
        }

        data["parameters"] = {k: v for k, v in data["parameters"].items() if v is not None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "output" in result and "results" in result["output"]:
                            image_url = result["output"]["results"][0]["url"]
                            return ImageGenerationResult(
                                success=True,
                                image_url=image_url
                            )
                        else:
                            error_msg = result.get("message", "未知错误")
                            return ImageGenerationResult(
                                success=False,
                                error_message=f"通义万相API错误: {error_msg}"
                            )
                    else:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("message", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}: {error_text}"
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"通义万相API错误: {error_msg}"
                        )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"通义万相请求异常: {str(e)}"
            )

    async def generate_image_edit(self, prompt: str, images: List[str], negative_prompt: Optional[str] = None) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        base_url = self.get_config_value("i2i_base_url", "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation")
        model = self.get_config_value("i2i_model", "wan2.6-i2i")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }

        parameters = {
            "n": 1,
            "prompt_extend": True,
            "watermark": False
        }

        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt

        data = {
            "model": model,
            "input": {
                "prompt": prompt,
                "images": images
            },
            "parameters": parameters
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()

                    if response.status != HTTPStatus.OK:
                        error_msg = result.get("message", f"HTTP {response.status}")
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"创建任务失败: {error_msg}"
                        )

                    if "output" not in result or "task_id" not in result["output"]:
                        return ImageGenerationResult(
                            success=False,
                            error_message="创建任务失败: 无效的响应格式"
                        )

                    task_id = result["output"]["task_id"]

                    return await self._wait_for_task_completion(task_id, api_key, session)

        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"通义万相图片编辑请求异常: {str(e)}"
            )

    async def _wait_for_task_completion(self, task_id: str, api_key: str, session: aiohttp.ClientSession) -> ImageGenerationResult:
        query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        max_attempts = 60
        attempt = 0

        while attempt < max_attempts:
            await asyncio.sleep(3)
            attempt += 1

            try:
                async with session.get(
                    query_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()

                    if response.status != HTTPStatus.OK:
                        error_msg = result.get("message", f"HTTP {response.status}")
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"查询任务失败: {error_msg}"
                        )

                    if "output" not in result:
                        continue

                    task_status = result["output"].get("task_status")

                    if task_status == "SUCCEEDED":
                        if "results" in result["output"] and len(result["output"]["results"]) > 0:
                            image_url = result["output"]["results"][0]["url"]
                            return ImageGenerationResult(
                                success=True,
                                image_url=image_url
                            )
                        else:
                            return ImageGenerationResult(
                                success=False,
                                error_message="任务完成但未返回图片"
                            )
                    elif task_status == "FAILED":
                        error_msg = result["output"].get("message", "任务失败")
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"图片生成失败: {error_msg}"
                        )

            except Exception as e:
                return ImageGenerationResult(
                    success=False,
                    error_message=f"查询任务异常: {str(e)}"
                )

        return ImageGenerationResult(
            success=False,
            error_message="任务超时: 等待时间超过3分钟"
        )
    
    def _map_size(self, width: int, height: int) -> str:
        """映射尺寸到通义万相支持的格式"""
        if width == height:
            if width <= 512:
                return "512*512"
            elif width <= 768:
                return "768*768"
            else:
                return "1024*1024"
        elif width > height:
            return "1280*720"
        else:
            return "720*1280"