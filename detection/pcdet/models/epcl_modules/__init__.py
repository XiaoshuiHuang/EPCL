from .epcl_detection import build_epcl

MODEL_FUNCS = {
    "EPCLDetection": build_epcl
}

def build_epcl_model(args):
    model = MODEL_FUNCS[args.NAME](args)
    return model