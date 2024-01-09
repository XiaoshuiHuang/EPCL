import yaml
from easydict import EasyDict

def get_network(args):

    if 'epcl' == args.arch:
        from .epcl import getEPCLSegNet as getNetwork
    elif 'pointtransformer' == args.arch:
        from .pointtransformer import getPointTransformerSegNet as getNetwork
        # elif 'pointnet' == args.arch:
        #     from .pointnet   import getPointMixerSegNet as getNetwork
    else:
        raise NotImplementedError
    
    kwargs = \
        {
            'intraLayer': args.intraLayer,
            'interLayer': args.interLayer,
            'transup': args.transup,
            'transdown': args.transdown,
            'stride': args.downsample,
        }
    model = getNetwork(c=args.fea_dim, k=args.classes, nsample=args.nsample, **kwargs)
    
    return model