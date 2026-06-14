python3 generate_test.py
nvcc -O3 -arch sm_89 bilaterial.cu -o bilaterial
./bilaterial test_input.bmp result_gpu.bmp result_cpu.bmp 1.5 25.0

