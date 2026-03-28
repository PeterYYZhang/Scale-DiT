from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type, Union


class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        if self.activated:
            return

        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(0)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]


class set_lora_scale:
    def __init__(self, lora_modules: List[BaseTunerLayer], scale: float) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]
        self.scale = scale

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(self.scale)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]


class enable_only_lora:
    def __init__(
        self,
        lora_modules: List[BaseTunerLayer],
        names: Optional[Union[str, List[str]]] = "default",
    ) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.allowed_names: List[str] = self._normalize_names(names)
        self.scales = [
            {
                adapter_name: lora_module.scaling.get(adapter_name, 1.0)
                for adapter_name in getattr(lora_module, "scaling", {}).keys()
            }
            for lora_module in self.lora_modules
        ]

    def _normalize_names(self, names: Optional[Union[str, List[str]]]) -> List[str]:
        if names is None:
            return ["default"]
        if isinstance(names, str):
            if "," in names:
                parsed = [n.strip() for n in names.split(",") if n.strip()]
                return parsed or ["default"]
            return [names]
        if not names:
            return ["default"]
        return list(names)

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for adapter_name in getattr(lora_module, "scaling", {}).keys():
                if adapter_name not in self.allowed_names:
                    lora_module.scaling[adapter_name] = 0

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for adapter_name, saved_scale in self.scales[i].items():
                lora_module.scaling[adapter_name] = saved_scale