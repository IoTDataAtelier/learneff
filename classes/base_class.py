from abc import ABC

class BaseClass(ABC):

    def set_attributes(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)
