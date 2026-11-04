"""Build an isolated activity collector from the installed CUDA headers."""
from .common import *


def main():
    initialize()
    directory = ROOT / 'tools'
    directory.mkdir(exist_ok=True)
    source = SOURCE / 'kernel_activity.cc'
    sha = digest(source)
    output = directory / f'kernel_activity_{sha[:16]}.so'
    command = ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-pthread',
               '-I/usr/local/cuda-12.8/targets/x86_64-linux/include', str(source),
               '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
               '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib',
               '-lcupti', '-o', str(output)]
    if not output.exists():
        subprocess.run(command, check=True)
    write(directory / 'activity_build.json', dict(command=command, source_SHA256=sha,
          library=str(output), library_SHA256=digest(output),
          scope='Prepared profiler alternative. GPU coverage and overhead are not yet validated.'))


if __name__ == '__main__':
    main()
