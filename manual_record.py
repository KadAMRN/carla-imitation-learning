import multiprocessing
import subprocess

def run_script(script_path):
    subprocess.call(["python", script_path])

# Specify the paths of your two scripts
script1_path = "C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\server.py"
script2_path = "C:\\Users\Kader\\OneDrive\Bureau\\WindowsNoEditor\\PythonAPI\\examples\\manual_control.py"

if __name__ == '__main__':
    # Create two processes for running the scripts
    process1 = multiprocessing.Process(target=run_script, args=(script1_path,))
    process2 = multiprocessing.Process(target=run_script, args=(script2_path,))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()