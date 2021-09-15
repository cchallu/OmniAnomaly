source venv/bin/activate

machines=('machine-3-1'  'machine-3-2'  'machine-3-3' 'machine-3-4' 'machine-3-5' 'machine-3-6' 'machine-3-7' 'machine-3-8' 'machine-3-9' 'machine-3-10' 'machine-3-11')
 
# Print array values in  lines
for val in ${machines[*]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py --n_intervals 5 --occlusion_prob 0.5 --dataset $val
done
