import machine
import sys
import utime

adc = machine.ADC(26)
s_pins = [machine.Pin(i, machine.Pin.OUT) for i in range(2, 6)]

def stream_boundary_tensor():
    while True:
        tensor = []
        for i in range(16):
            for j in range(4): # Set MUX 4-bit binary address
                s_pins[j].value((i >> j) & 1)
            utime.sleep_us(10) # Settling time for multiplexer
            tensor.append(adc.read_u16())
        # Stream comma-separated tensor to the Host computer
        sys.stdout.write(','.join(map(str, tensor)) + '\n')
        utime.sleep_ms(2) # Yield for ~500Hz full-array sampling

if __name__ == "__main__":
    stream_boundary_tensor()
