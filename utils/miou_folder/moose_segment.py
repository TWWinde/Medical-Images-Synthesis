from moosez import moose

model_name = 'clin_ct_organs'
input_dir = '/home/Documents/your_project/data/input'
output_dir = '/home/Documents/your_project/data/output'
accelerator = 'cuda'
moose(model_name, input_dir, output_dir, accelerator)