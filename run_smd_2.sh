source venv/bin/activate

machines=('machine-2-1'  'machine-2-2'  'machine-2-3' 'machine-2-4' 'machine-2-5' 'machine-2-6' 'machine-2-7' 'machine-2-8' 'machine-2-9')
 
# Print array values in  lines
for val in ${machines[*]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py --n_intervals 5 --occlusion_prob 0.5 --dataset $val
done
