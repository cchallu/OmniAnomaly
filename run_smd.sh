source venv/bin/activate

machines=('machine-1-1'  'machine-1-2'  'machine-1-3' 'machine-1-4' 'machine-1-5' 'machine-1-6' 'machine-1-7' 'machine-1-8')
 
# Print array values in  lines
for val in ${machines[*]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py --n_intervals 5 --occlusion_prob 0.5 --dataset $val
done
