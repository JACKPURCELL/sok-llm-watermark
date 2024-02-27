from .john23.john23 import john23_WatermarkLogitsProcessor, john23_WatermarkDetector
from .rohith23.rohith23 import rohith23_WatermarkLogitsProcessor, rohith23_WatermarkDetector
from .xuandong23 import xuandong23_WatermarkLogitsProcessor
from .xuandong23b import xuandong23b_WatermarkLogitsProcessor, xuandong23b_WatermarkDetector
from .lean23.lean23 import lean23_BalanceMarkingWatermarkLogitsProcessor, lean23_WatermarkDetector
from .aiwei23.aiwei23 import aiwei23_WatermarkLogitsProcessor, aiwei23_WatermarkDetector, prepare_generator
from .kiyoon23.kiyoon23 import kiyoon23
from .xiaoniu23.unbiased_watermark import Delta_Reweight, Gamma_Reweight, PrevN_ContextCodeExtractor, patch_model, WatermarkLogitsProcessor as xiaoniu23_WatermarkLogitsProcessor
from .xiaoniu23.xiaoniu23 import xiaoniu23_detector, generate_with_watermark as generate_with_watermark_xiaoniu23
from .aiwei23b.aiwei23b import prepare_watermark_model as aiwei23b_prepare_watermark_model, watermark as aiwei23b_watermark, aiwei23b_WatermarkDetector
from .aiwei23b.watermark import WatermarkLogitsProcessor as aiwei23b_WatermarkLogitsProcessor

from .christ23.christ23 import christ23_WatermarkLogitsProcessor, christ23_WatermarkDetector


    ######################################################################
    # Add your code here
    ######################################################################
    # If you have new watermark, add them here
    ######################################################################   