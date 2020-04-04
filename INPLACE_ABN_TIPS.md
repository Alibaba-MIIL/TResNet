## Some Tips For Working With Inplace-ABN

[Inplace-ABN](https://github.com/mapillary/inplace_abn) have exactly the same fields as
[regular BatchNorm](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py):
* module.weight
* module.bias
* module.running_mean
* module.running_var
* module.num_batches_tracked

Therefore, any function that operates on BatchNorm can run on
Inplace-ABN.

However, problems can arise when a logic condition seeks explicitly for
BatchNorm layers only:
```
if isinstance(module, nn.BatchNorm2d): 
    do_something(module)
```

Anywhere you see a code segement like this, it needs to be replaced with
a condition that include Inplace-ABN:
```
if isinstance(module, nn.BatchNorm2d) or isinstance(module, inplace_abn.InPlaceABN): 
    do_something(module)
```

##### NVIDIA Apex mixed precision
[NVIDIA Apex](https://github.com/NVIDIA/apex) O0, O1 and O3
mixed-precision options work seemlesly on Inplace-ABN. 

For O2 mixed precision, we need to convert manually Inplace-ABN to fp32,
since NVIDIA Apex inner code has explicit 'if' condition for BatchNorm
only. 

Conversion can be done easily with the helper function
'IABN2float':
```
if args.use_apex: 
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.opt_level) 
    if args.opt_level == 'O2': # IABN needs adjustment for O2 
        from src.models.tresnet import IABN2float
        model = IABN2float(model)
```