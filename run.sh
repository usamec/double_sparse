for sp in 0.5 0.6 0.7; do
   for fm in "" "--fix-mask"; do 
      for final in "" "--no-final"; do
        echo llama2-13-$sp$fm$final;
        python llama.py meta-llama/Llama-2-13b-hf c4 --sparsity $sp $fm $final | tee logs/llama2-13-$sp$fm$final; 
      done
    done
done
