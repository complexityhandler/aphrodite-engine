import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from aphrodite.common.config import (CacheConfig, DecodingConfig, DeviceConfig,
                                     EngineConfig, LoadConfig, LoRAConfig,
                                     ModelConfig, ParallelConfig,
                                     SchedulerConfig, SpeculativeConfig,
                                     TokenizerPoolConfig, VisionLanguageConfig)
from aphrodite.common.utils import is_cpu, str_to_int_tuple
from aphrodite.quantization import QUANTIZATION_METHODS


@dataclass
class EngineArgs:
    """Arguments for Aphrodite engine."""

    model: str
    served_model_name: Optional[Union[List[str]]] = None
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quantization_param_path: Optional[str] = None
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: Optional[bool] = False
    distributed_executor_backend: Optional[str] = None
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = False
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 10  # OpenAI default is 5, setting to 10 because ST
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_smooth: bool = False
    deepspeed_fp_bits: Optional[int] = None
    enforce_eager: bool = True
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    tokenizer_pool_type: str = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: str = "auto"
    max_cpu_loras: Optional[int] = None
    device: str = "auto"
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    preemption_mode: Optional[str] = None
    # Related to Vision-language models such as llava
    image_token_id: Optional[int] = None
    image_input_shape: Optional[str] = None
    image_feature_size: Optional[int] = None
    # Scheduler config
    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: bool = False
    guided_decoding_backend: str = 'outlines'
    # Speculative decoding config
    speculative_model: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None

    qlora_adapter_name_or_path: Optional[str] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        if is_cpu():
            self.distributed_executor_backend = None

    @staticmethod
    def add_cli_args_for_vlm(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--image-token-id',
                            type=int,
                            default=None,
                            help=('Input id for image token.'))
        parser.add_argument(
            '--image-input-shape',
            type=str,
            default=None,
            help=('The biggest image input shape (worst for memory footprint) '
                  'given an input type. Only used for the profile_run.'))
        parser.add_argument(
            '--image-feature-size',
            type=int,
            default=None,
            help=('The image feature size along the context dimension.'))

        return parser

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for the Aphrodite engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            "--model",
            type=str,
            default="EleutherAI/pythia-70m-deduped",
            help="name or path of the huggingface model to use",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=EngineArgs.tokenizer,
            help="name or path of the huggingface tokenizer to use",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="Skip initialization of tokenizer and detokenizer")
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="the specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--code-revision",
            type=str,
            default=None,
            help="the specific revision to use for the model code on "
            "Hugging Face Hub. It can be a branch name, a tag name, or a "
            "commit id. If unspecified, will use the default version.",
        )
        parser.add_argument(
            "--tokenizer-revision",
            type=str,
            default=None,
            help="the specific tokenizer version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help='tokenizer mode. "auto" will use the fast '
            'tokenizer if available, and "slow" will '
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="trust remote code from huggingface",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=EngineArgs.download_dir,
            help="directory to download and load the weights, "
            "default to the default cache dir of "
            "huggingface",
        )
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[
                'auto',
                'pt',
                'safetensors',
                'npcache',
                'dummy',
                'tensorizer',
                'sharded_state',
                'bitsandbytes',
            ],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize Aphrodite Model script in the '
            'Examples section for more information.\n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
        parser.add_argument(
            '--quantization-param-path',
            type=str,
            default=None,
            help='Path to the JSON file containing the KV cache '
            'scaling factors. This should generally be supplied, when '
            'KV cache dtype is FP8. Otherwise, KV cache scaling factors '
            'default to 1.0, which may cause accuracy issues. '
            'FP8_E5M2 (without scaling) is only supported on cuda version'
            'greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead '
            'supported for common inference criteria. ')
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="model context length. If unspecified, "
            "will be automatically derived from the model.",
        )
        parser.add_argument(
            '--guided-decoding-backend',
            type=str,
            default='outlines',
            choices=['outlines', 'lm-format-enforcer'],
            help='Which engine will be used for guided decoding'
            ' (JSON schema / regex etc) by default. Currently support '
            'https://github.com/outlines-dev/outlines and '
            'https://github.com/noamgat/lm-format-enforcer.'
            ' Can be overridden per request via guided_decoding_backend'
            ' parameter.')
        # Parallel arguments
        parser.add_argument(
            '--distributed-executor-backend',
            choices=['ray', 'mp'],
            default=EngineArgs.distributed_executor_backend,
            help='Backend to use for distributed serving. When more than 1 GPU '
            'is used, will be automatically set to "ray" if installed '
            'or "mp" (multiprocessing) otherwise.')
        parser.add_argument(
            '--worker-use-ray',
            action='store_true',
            help='Deprecated, use --distributed-executor-backend=ray.')
        parser.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=EngineArgs.pipeline_parallel_size,
            help="number of pipeline stages. Currently not supported.")
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="number of tensor parallel replicas, i.e. the number of GPUs "
            "to use.")
        parser.add_argument(
            "--max-parallel-loading-workers",
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help="load model sequentially in multiple batches, "
            "to avoid RAM OOM when using tensor "
            "parallel and large models",
        )
        parser.add_argument(
            "--ray-workers-use-nsight",
            action="store_true",
            help="If specified, use nsight to profile ray workers",
        )
        # KV cache arguments
        parser.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            choices=[8, 16, 32],
            help="token block size",
        )
        parser.add_argument(
            "--enable-prefix-caching",
            "--context-shift",
            action="store_true",
            help="Enable automatic prefix caching.",
        )
        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size')
        parser.add_argument("--use-v2-block-manager",
                            action="store_true",
                            help="Use the v2 block manager.")
        parser.add_argument(
            "--num-lookahead-slots",
            type=int,
            default=EngineArgs.num_lookahead_slots,
            help="Experimental scheduling config necessary for "
            "speculative decoding. This will be replaced by "
            "speculative decoding config in the future; it is "
            "present for testing purposes until then.")
        parser.add_argument("--seed",
                            type=int,
                            default=EngineArgs.seed,
                            help="random seed")
        parser.add_argument(
            "--swap-space",
            type=int,
            default=EngineArgs.swap_space,
            help="CPU swap space size (GiB) per GPU",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            "-gmu",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="the fraction of GPU memory to be used for "
            "the model executor, which can range from 0 to 1."
            "If unspecified, will use the default value of 0.9.",
        )
        parser.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=None,
            help="If specified, ignore GPU profiling result and use this "
            "number of GPU blocks. Used for testing preemption.")
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="maximum number of batched tokens per "
            "iteration",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="maximum number of sequences per iteration",
        )
        parser.add_argument(
            "--max-logprobs",
            type=int,
            default=EngineArgs.max_logprobs,
            help="maximum number of log probabilities to "
            "return.",
        )
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="disable logging statistics",
        )
        # Quantization settings.
        parser.add_argument(
            "--quantization",
            "-q",
            type=str,
            choices=[*QUANTIZATION_METHODS, None],
            default=EngineArgs.quantization,
            help="Method used to quantize the weights. If "
            "None, we first check the `quantization_config` "
            "attribute in the model config file. If that is "
            "None, we assume the model weights are not "
            "quantized and use `dtype` to determine the data "
            "type of the weights.",
        )
        parser.add_argument(
            "--load-in-4bit",
            action="store_true",
            help="Load the FP16 model in 4-bit format. Also "
            "works with AWQ models. Throughput at 2.5x of "
            "FP16.",
        )
        parser.add_argument(
            "--load-in-8bit",
            action="store_true",
            help="Load the FP16 model in 8-bit format. "
            "Throughput at 0.3x of FP16.",
        )
        parser.add_argument(
            "--load-in-smooth",
            action="store_true",
            help="Load the FP16 model in smoothquant "
            "8bit format. Throughput at 0.7x of FP16. ",
        )
        parser.add_argument(
            "--deepspeed-fp-bits",
            type=int,
            default=None,
            help="Number of floating bits to use for the deepseed "
            "quantization. Supported bits are: 4, 6, 8, 12. ")
        parser.add_argument('--rope-scaling',
                            default=None,
                            type=json.loads,
                            help='RoPE scaling configuration in JSON format. '
                            'For example, {"type":"dynamic","factor":2.0}')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
        parser.add_argument(
            "--enforce-eager",
            type=lambda x: (str(x).lower() == 'true'),
            default=EngineArgs.enforce_eager,
            help="Always use eager-mode PyTorch. If False, "
            "will use eager mode and CUDA graph in hybrid "
            "for maximal performance and flexibility.",
        )
        parser.add_argument("--max-context-len-to-capture",
                            type=int,
                            default=EngineArgs.max_context_len_to_capture,
                            help="Maximum context length covered by CUDA "
                            "graphs. When a sequence has context length "
                            "larger than this, we fall back to eager mode. "
                            "(DEPRECATED. Use --max-seq_len-to-capture instead"
                            ")")
        parser.add_argument("--max-seq_len-to-capture",
                            type=int,
                            default=EngineArgs.max_seq_len_to_capture,
                            help="Maximum sequence length covered by CUDA "
                            "graphs. When a sequence has context length "
                            "larger than this, we fall back to eager mode.")
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=EngineArgs.disable_custom_all_reduce,
            help="See ParallelConfig",
        )
        parser.add_argument("--tokenizer-pool-size",
                            type=int,
                            default=EngineArgs.tokenizer_pool_size,
                            help="Size of tokenizer pool to use for "
                            "asynchronous tokenization. If 0, will "
                            "use synchronous tokenization.")
        parser.add_argument("--tokenizer-pool-type",
                            type=str,
                            default=EngineArgs.tokenizer_pool_type,
                            help="The type of tokenizer pool to use for "
                            "asynchronous tokenization. Ignored if "
                            "tokenizer_pool_size is 0.")
        parser.add_argument("--tokenizer-pool-extra-config",
                            type=str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help="Extra config for tokenizer pool. "
                            "This should be a JSON string that will be "
                            "parsed into a dictionary. Ignored if "
                            "tokenizer_pool_size is 0.")
        parser.add_argument(
            '--preemption-mode',
            type=str,
            default=None,
            help='If \'recompute\', the engine performs preemption by block '
            'swapping; If \'swap\', the engine performs preemption by block '
            'swapping.')
        # LoRA related configs
        parser.add_argument(
            "--enable-lora",
            action="store_true",
            help="If True, enable handling of LoRA adapters.",
        )
        parser.add_argument(
            "--max-loras",
            type=int,
            default=EngineArgs.max_loras,
            help="Max number of LoRAs in a single batch.",
        )
        parser.add_argument(
            "--max-lora-rank",
            type=int,
            default=EngineArgs.max_lora_rank,
            help="Max LoRA rank.",
        )
        parser.add_argument(
            "--lora-extra-vocab-size",
            type=int,
            default=EngineArgs.lora_extra_vocab_size,
            help=("Maximum size of extra vocabulary that can be "
                  "present in a LoRA adapter (added to the base "
                  "model vocabulary)."),
        )
        parser.add_argument(
            "--lora-dtype",
            type=str,
            default=EngineArgs.lora_dtype,
            choices=["auto", "float16", "bfloat16", "float32"],
            help=("Data type for LoRA. If auto, will default to "
                  "base model dtype."),
        )
        parser.add_argument(
            "--long-lora-scaling-factors",
            type=str,
            default=EngineArgs.long_lora_scaling_factors,
            help=("Specify multiple scaling factors (which can "
                  "be different from base model scaling factor "
                  "- see eg. Long LoRA) to allow for multiple "
                  "LoRA adapters trained with those scaling "
                  "factors to be used at the same time. If not "
                  "specified, only adapters trained with the "
                  "base model scaling factor are allowed."))
        parser.add_argument(
            "--max-cpu-loras",
            type=int,
            default=EngineArgs.max_cpu_loras,
            help=("Maximum number of LoRAs to store in CPU memory. "
                  "Must be >= than max_num_seqs. "
                  "Defaults to max_num_seqs."),
        )
        parser.add_argument(
            "--fully-sharded-loras",
            action='store_true',
            help=("By default, only half of the LoRA computation is sharded "
                  "with tensor parallelism. Enabling this will use the fully "
                  "sharded layers. At high sequence length, max rank or "
                  "tensor parallel size, this is likely faster."))
        parser.add_argument(
            "--device",
            type=str,
            default=EngineArgs.device,
            choices=[
                "auto", "cuda", "neuron", "cpu", "openvino", "tpu", "xpu"
            ],
            help=("Device to use for model execution."),
        )
        # Related to Vision-language models such as llava
        parser = EngineArgs.add_cli_args_for_vlm(parser)
        parser.add_argument(
            "--scheduler-delay-factor",
            "-sdf",
            type=float,
            default=EngineArgs.scheduler_delay_factor,
            help="Apply a delay (of delay factor multiplied by previous "
            "prompt latency) before scheduling next prompt.")
        parser.add_argument(
            "--enable-chunked-prefill",
            action="store_true",
            help="If True, the prefill requests can be chunked based on the "
            "max_num_batched_tokens.")
        parser.add_argument(
            "--speculative-model",
            type=str,
            default=EngineArgs.speculative_model,
            help=
            "The name of the draft model to be used in speculative decoding.")
        parser.add_argument(
            "--speculative-draft-tensor-parallel-size",
            "-spec-draft-tp",
            type=int,
            default=EngineArgs.speculative_draft_tensor_parallel_size,
            help="Number of tensor parallel replicas for "
            "the draft model in speculative decoding.")
        parser.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=EngineArgs.num_speculative_tokens,
            help="The number of speculative tokens to sample from "
            "the draft model in speculative decoding")
        parser.add_argument(
            "--speculative-max-model-len",
            type=str,
            default=EngineArgs.speculative_max_model_len,
            help="The maximum sequence length supported by the "
            "draft model. Sequences over this length will skip "
            "speculation.")
        parser.add_argument(
            "--speculative-disable-by-batch-size",
            type=int,
            default=EngineArgs.speculative_disable_by_batch_size,
            help="Disable speculative decoding for new incoming requests "
            "if the number of enqueue requests is larger than this value.")
        parser.add_argument(
            "--ngram-prompt-lookup-max",
            type=int,
            default=EngineArgs.ngram_prompt_lookup_max,
            help="Max size of window for ngram prompt lookup in speculative "
            "decoding.")
        parser.add_argument(
            "--ngram-prompt-lookup-min",
            type=int,
            default=EngineArgs.ngram_prompt_lookup_min,
            help="Min size of window for ngram prompt lookup in speculative "
            "decoding.")

        parser.add_argument(
            '--spec-decoding-acceptance-method',
            type=str,
            default=EngineArgs.spec_decoding_acceptance_method,
            choices=['rejection_sampler', 'typical_acceptance_sampler'],
            help='Specify the acceptance method to use during draft token '
            'verification in speculative decoding. Two types of acceptance '
            'routines are supported: '
            '1) RejectionSampler which does not allow changing the '
            'acceptance rate of draft tokens, '
            '2) TypicalAcceptanceSampler which is configurable, allowing for '
            'a higher acceptance rate at the cost of lower quality, '
            'and vice versa.')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-threshold',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_threshold,
            help='Set the lower bound threshold for the posterior '
            'probability of a token to be accepted. This threshold is '
            'used by the TypicalAcceptanceSampler to make sampling decisions '
            'during speculative decoding. Defaults to 0.09')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-alpha',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_alpha,
            help='A scaling factor for the entropy-based threshold for token '
            'acceptance in the TypicalAcceptanceSampler. Typically defaults '
            'to sqrt of --typical-acceptance-sampler-posterior-threshold '
            'i.e. 0.3')
        parser.add_argument("--model-loader-extra-config",
                            type=str,
                            default=EngineArgs.model_loader_extra_config,
                            help="Extra config for model loader. "
                            "This will be passed to the model loader "
                            "corresponding to the chosen load_format. "
                            "This should be a JSON string that will be "
                            "parsed into a dictionary.")
        parser.add_argument(
            "--served-model-name",
            nargs="+",
            type=str,
            default=None,
            help="The model name(s) used in the API. If multiple "
            "names are provided, the server will respond to any "
            "of the provided names. The model name in the model "
            "field of a response will be the first name in this "
            "list. If not specified, the model name will be the "
            "same as the `--model` argument. Noted that this name(s)"
            "will also be used in `model_name` tag content of "
            "prometheus metrics, if multiple names provided, metrics"
            "tag will take the first one.")
        parser.add_argument("--qlora-adapter-name-or-path",
                            type=str,
                            default=None,
                            help="Name or path of the LoRA adapter to use.")

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(self, ) -> EngineConfig:

        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        if (self.quantization == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.load_format != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes quantization and QLoRA adapter only support "
                f"'bitsandbytes' load format, but got {self.load_format}")

        if (self.load_format == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes load format and QLoRA adapter only support "
                f"'bitsandbytes' quantization, but got {self.quantization}")
        if self.image_token_id is not None:
            if (not self.image_input_shape or not self.image_feature_size):
                raise ValueError(
                    'Specify `image_input_shape` and '
                    '`image_feature_size` together with `image_token_id`.')

            vision_language_config = VisionLanguageConfig(
                image_token_id=self.image_token_id,
                image_input_shape=str_to_int_tuple(self.image_input_shape),
                image_feature_size=self.image_feature_size,
            )
        else:
            vision_language_config = None

        device_config = DeviceConfig(device=self.device)

        model_config = ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            load_in_smooth=self.load_in_smooth,
            deepspeed_fp_bits=self.deepspeed_fp_bits,
            quantization_param_path=self.quantization_param_path,
            enforce_eager=self.enforce_eager,
            max_context_len_to_capture=self.max_context_len_to_capture,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            multimodal_config=vision_language_config,
        )

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
        )

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            worker_use_ray=self.worker_use_ray,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            tokenizer_pool_config=TokenizerPoolConfig.create_config(
                tokenizer_pool_size=self.tokenizer_pool_size,
                tokenizer_pool_type=self.tokenizer_pool_type,
                tokenizer_pool_extra_config=self.tokenizer_pool_extra_config,
            ),
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            distributed_executor_backend=self.distributed_executor_backend)

        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            speculative_draft_tensor_parallel_size=self.
            speculative_draft_tensor_parallel_size,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_disable_by_batch_size=self.
            speculative_disable_by_batch_size,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            use_v2_block_manager=self.use_v2_block_manager,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=self.
            typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=self.
            typical_acceptance_sampler_posterior_alpha,
        )

        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            use_v2_block_manager=self.use_v2_block_manager,
            num_lookahead_slots=(self.num_lookahead_slots
                                 if speculative_config is None else
                                 speculative_config.num_lookahead_slots),
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            embedding_mode=model_config.embedding_mode,
            preemption_mode=self.preemption_mode,
        )

        lora_config = LoRAConfig(
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_extra_vocab_size=self.lora_extra_vocab_size,
            long_lora_scaling_factors=self.long_lora_scaling_factors,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras
            and self.max_cpu_loras > 0 else None) if self.enable_lora else None

        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            if self.model_loader_extra_config is None:
                self.model_loader_extra_config = {}
            self.model_loader_extra_config[
                "qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
        )

        decoding_config = DecodingConfig(
            guided_decoding_backend=self.guided_decoding_backend)

        if (model_config.get_sliding_window() is not None
                and scheduler_config.chunked_prefill_enabled
                and not scheduler_config.use_v2_block_manager):
            raise ValueError(
                "Chunked prefill is not supported with sliding window. "
                "Set --disable-sliding-window to disable sliding window.")

        return EngineConfig(model_config=model_config,
                            cache_config=cache_config,
                            parallel_config=parallel_config,
                            scheduler_config=scheduler_config,
                            device_config=device_config,
                            lora_config=lora_config,
                            vision_language_config=vision_language_config,
                            speculative_config=speculative_config,
                            load_config=load_config,
                            decoding_config=decoding_config)


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous Aphrodite engine."""

    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: int = 0
    uvloop: bool = False

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--engine-use-ray",
            action="store_true",
            help="use Ray to start the LLM engine in a "
            "separate process as the server process.",
        )
        parser.add_argument(
            "--disable-log-requests",
            action="store_true",
            help="disable logging requests",
        )
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=0,
            help="max number of prompt characters or prompt "
            "ID numbers being printed in log. "
            "Default: unlimited.",
        )
        parser.add_argument(
            "--uvloop",
            action="store_true",
            help="Use the Uvloop asyncio event loop to possibly increase "
            "performance")
        return parser
