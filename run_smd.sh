#source venv/bin/activate

machines=('machine-1-1'  'machine-1-2'  'machine-1-3' 'machine-1-4' 'machine-1-5' 'machine-1-6' 'machine-1-7' 'machine-1-8' 'machine-2-1'  'machine-2-2'  'machine-2-3' 'machine-2-4' 'machine-2-5' 'machine-2-6' 'machine-2-7' 'machine-2-8' 'machine-2-9' 'machine-3-1'  'machine-3-2'  'machine-3-3' 'machine-3-4' 'machine-3-5' 'machine-3-6' 'machine-3-7' 'machine-3-8' 'machine-3-9' 'machine-3-10' 'machine-3-11')
 
# Print array values in  lines
for val in ${machines[*]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py --n_intervals 5 --occlusion_prob 0 --dataset $val
done
