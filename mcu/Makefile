# Host build (test harness) and a Cortex-M4F build of the library objects (to measure flash/RAM with real Arm code).
#   make            -> build/kws_test (laptop)
#   make cortex-m4  -> build/cm4/*.o + size report (clang --target=thumbv7em, -Os; STM32F4/L4 class core)
#   make cortex-m7  -> same for Cortex-M7 (STM32F7/H7)
# The generated/ headers come from: python -I export_mcu.py runs/<task>/int8.pkl
SRC = dscnn_int8.c kws_frontend.c generated/kws_model_data.c generated/kws_frontend_tables.c
CFLAGS = -std=c99 -O2 -Wall -Wextra -I. -Igenerated

all: build/kws_test

build/kws_test: $(SRC) kws_test.c dscnn_int8.h kws_frontend.h
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) kws_test.c -lm

ARM_FLAGS = -std=c99 -Os -ffreestanding -fno-common -ffunction-sections -fdata-sections -I. -Igenerated -Wall

cortex-m4:
	mkdir -p build/cm4
	for f in $(SRC); do clang --target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
	  $(ARM_FLAGS) -c $$f -o build/cm4/$$(basename $${f%.c}).o || exit 1; done
	llvm-size -A build/cm4/*.o | grep -E "^build|\.text|\.rodata|\.bss|\.data|^Total"

cortex-m7:
	mkdir -p build/cm7
	for f in $(SRC); do clang --target=thumbv7em-none-eabihf -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 \
	  $(ARM_FLAGS) -c $$f -o build/cm7/$$(basename $${f%.c}).o || exit 1; done
	llvm-size -A build/cm7/*.o | grep -E "^build|\.text|\.rodata|\.bss|\.data|^Total"

clean:
	rm -rf build
