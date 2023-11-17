from moosez import moose

model_name = 'clin_ct_organs'
input_dir = '/misc/no_backups/s1449/Medical-Images-Synthesis/results/medicals/MOOSEv2_data'
output_dir = '/misc/no_backups/s1449/Medical-Images-Synthesis/results/medicals/moose_segment_results'
accelerator = 'cuda'
moose(model_name, input_dir, output_dir, accelerator)