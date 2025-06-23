nvcc sgemm.cu -lcublas -o v0  
ncu --set full --target-processes all -o my_report ./v3