"""Official example EC transport with explicit consumer-cache validation."""
from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnector


class StrictExampleConnector(ECExampleConnector):
    def has_cache_item(self, identifier):
        found=super().has_cache_item(identifier)
        if self.is_consumer and not found:
            raise FileNotFoundError('Missing required online E export for image identifier '+identifier+
                '. Local vision fallback is forbidden for the EPD baseline.')
        return found
