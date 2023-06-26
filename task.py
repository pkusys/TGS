import subprocess
import os
from time import time
import utils


class JobInfo(object):
    def __init__(self, job_id, job_name, batch_size, iterations, num_gpus, priority, thread_percentage, image_name, antman_config, antman_status) -> None:
        super().__init__()
        assert num_gpus <= 2

        self.job_id = job_id
        self.job_name = job_name
        self.batch_size = batch_size
        self.iterations = iterations
        self.num_gpus = num_gpus
        self.gpus = '' #','.join([str(i) for i in range(num_gpus)])
        self.priority = priority
        self.thread_percentage = thread_percentage
        self.image_name = image_name
        self.antman_config = antman_config
        self.antman_status = antman_status


class Task(object):
    def __init__(self, job_info: JobInfo, scheduler_ip, vcuda_mounts: dict, need_throughput) -> None:
        super().__init__()

        self._job_id = job_info.job_id
        self._job_name = job_info.job_name
        self._batch_size = job_info.batch_size
        self._iterations = job_info.iterations
        self._finished_iterations = 0
        self._gpus = job_info.gpus
        self._priority = job_info.priority
        self._thread_percentage = job_info.thread_percentage
        self.image_name = job_info.image_name
        self._scheduler_ip = scheduler_ip
        self._vcuda_mounts = vcuda_mounts
        self.need_throughput = need_throughput
        self.throughputs = list()
        self._timestamp = None
        self.last_time = time()
        self._antman_config = job_info.antman_config
        self._antman_status = job_info.antman_status
        self._idle_port = None
    

    def get_idle_port(self):
        if self._idle_port == None:
            self._idle_port = utils.find_free_port()
        return self._idle_port
    

    def mounts(self, additional_mounts: list):
        mounts = []
        for docker_mount in additional_mounts:
            mounts += ['-v', docker_mount]
        
        if self._priority in self._vcuda_mounts:
            for docker_mount in self._vcuda_mounts[self._priority]:
                mounts += ['-v', docker_mount]
        return mounts


    @staticmethod
    def test_kill_restart():
        bash_cmd = 'nvidia-smi; sleep 2m; date'
        return bash_cmd
    

    def pygcn(self):
        bash_cmd = f'python /cluster/workloads/pygcn/pygcn/train.py --epochs {self._iterations}'
        bash_cmd += f' --scheduler_ip {self._scheduler_ip}'
        bash_cmd += f' --trainer_port {self.get_idle_port()}'
        bash_cmd += f' --job_id {self._job_id}'
        return bash_cmd


    def bert(self):
        bash_cmd = f'python /cluster/workloads/run_squad.py --max_steps {self._iterations} --model_type=bert --model_name_or_path=/cluster/datasets/bert-config --do_train --do_lower_case --train_file=/cluster/datasets/squad-data/train-v2.0.json --per_gpu_train_batch_size {self._batch_size} --output_dir nlp_output --overwrite_output_dir --save_steps 0'
        bash_cmd += f' --scheduler_ip {self._scheduler_ip}'
        bash_cmd += f' --trainer_port {self.get_idle_port()}'
        bash_cmd += f' --job_id {self._job_id}'
        return bash_cmd
    

    def dlrm(self):
        # FIXME
        if self._job_id == 'dlrm':
            bash_cmd = f'python /cluster/workloads/dlrm/dlrm_s_pytorch.py --mini-batch-size=2048 --test-mini-batch-size=16384 --test-num-workers=0 --num-batches={self._iterations} --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=17000000-17000000-17000000-20000000-20000000-20000000-20000000-20000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=10 --print-time --enable-profiling --use-gpu'
        elif self._job_id == 'dlrm1':
            bash_cmd = f'python /cluster/workloads/dlrm/dlrm_s_pytorch.py --mini-batch-size=256 --test-mini-batch-size=16384 --test-num-workers=0 --num-batches={self._iterations} --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=17000000-17000000-17000000-20000000-20000000-20000000-20000000-20000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=10 --print-time --enable-profiling --use-gpu'
        else:
            bash_cmd = f'python /cluster/workloads/dlrm/dlrm_s_pytorch.py --mini-batch-size={self._batch_size} --test-mini-batch-size={self._batch_size} --test-num-workers=0 --num-batches={self._iterations} --data-generation=random --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --use-gpu'
         # bash_cmd = f'python /cluster/workloads/dlrm/dlrm_s_pytorch.py --mini-batch-size=2048 --test-mini-batch-size=16384 --test-num-workers=0 --num-batches={self._iterations} --data-generation=random --arch-mlp-bot=512-512-32 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=32 --arch-embedding-size=10000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=10 --print-time --enable-profiling --use-gpu'
        bash_cmd += f' --scheduler_ip {self._scheduler_ip}'
        bash_cmd += f' --trainer_port {self.get_idle_port()}'
        bash_cmd += f' --job_id {self._job_id}'
        return bash_cmd


    def imagenet(self):
        num_gpus = len(self._gpus.split(','))
        bash_cmd = ""
        if num_gpus == 1:
            bash_cmd = f'python /cluster/workloads/pytorch_imagenet_torchvision.py --iterations {self._iterations} --batch-size {self._batch_size} --model {self._job_name} --train-dir /cluster/datasets/tiny-imagenet-200/train'
        else:
            bash_cmd = f'horovodrun -np {num_gpus} -H localhost:{num_gpus} python /cluster/workloads/horovod_imagenet_torchvision.py --iterations {self._iterations} --batch-size {self._batch_size} --model {self._job_name} --train-dir /cluster/datasets/tiny-imagenet-200/train'
        bash_cmd += f' --scheduler_ip {self._scheduler_ip}'
        bash_cmd += f' --trainer_port {self.get_idle_port()}'
        bash_cmd += f' --job_id {self._job_id}'
        if num_gpus > 1:
            bash_cmd += ' |& grep -v "Read -1"'
        return bash_cmd


    def espnet2(self):
        bash_cmd = f'cd /workspace/espnet/egs2/aishell/asr1; ./run.sh --lm_args "--lm_conf layer=48"'
        return bash_cmd


    def tf_benchmarks(self, model_name):
        bash_cmd = f'python /cluster/workloads/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=1 --batch_size={self._batch_size} --model={model_name} --variable_update=replicated  --num_batches={self._iterations} --display_every=200 --data_name=imagenet --allow_growth' # --data_dir=/cluster/datasets/imagenet-tfrecord/train/'
        return bash_cmd


    def tf_gcn(self):
        bash_cmd = f'python /cluster/workloads/gcn/train.py --epochs {self._iterations}'
        return bash_cmd


    def tf_shufflenet(self):
        bash_cmd = f'python /cluster/workloads/shufflenet/main.py --batch-size {self._batch_size} --num-epochs {self._iterations} --config /cluster/workloads/shufflenet/config/test.json'
        return bash_cmd
    

    def tf_resnet_eager(self):
        bash_cmd = f'python /cluster/workloads/tf-eager-examples-master/scripts/05_resnet.py --batch-size {self._batch_size} --epochs {self._iterations}'
        return bash_cmd

    def megatron_gpt(self):
        bash_cmd = f'python benchmark_gpt_bert.py --nproc_per_node 2 --suite gpt.tmp --g_batch_size {self._batch_size} --repeat {self._iterations}'
        bash_cmd += f' --scheduler_ip {self._scheduler_ip}'
        bash_cmd += f' --trainer_port {self.get_idle_port()}'
        bash_cmd += f' --job_id {self._job_id}'
        return bash_cmd


    def run(self, mount: list):
        bash_cmd = ''
        if self._job_name == 'test_kill_restart':
            bash_cmd = self.test_kill_restart()
        elif self._job_name == 'gcn':
            bash_cmd = self.pygcn()
        elif self._job_name == 'bert':
            bash_cmd = self.bert()
        elif self._job_name[:4] == 'dlrm':
            bash_cmd = self.dlrm()
        elif self._job_name == 'espnet2':
            bash_cmd = self.espnet2()
        elif self._job_name in ['resnet50', 'resnet152', 'mobilenet_v2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x2_0', 'resnet34', 'alexnet']:
            bash_cmd = self.imagenet()
        elif self._job_name[:14] == 'tf_benchmarks-':
            bash_cmd = self.tf_benchmarks(self._job_name[14:])
        elif self._job_name == 'tf-gcn':
            bash_cmd = self.tf_gcn()
        elif self._job_name == 'tf-shufflenet':
            bash_cmd = self.tf_shufflenet()
        elif self._job_name == 'megatron-gpt':
            bash_cmd = self.megatron_gpt()
        # elif self._job_name == 'tf-resnet-eager':
        #     bash_cmd = self.tf_resnet_eager()
        else:
            raise Exception('wrong model name')
        if self._priority == 'high':
            if self._job_name == 'megatron-gpt':
                bash_cmd = 'cd /cluster/workloads/megatron/ && ' + 'nice -n -20 ' + bash_cmd
            else:
                bash_cmd = 'nice -n -20 ' + bash_cmd
        else:
            if self._job_name == 'megatron-gpt':
                bash_cmd = 'cd /cluster/workloads/megatron/ && ' + bash_cmd
        # elif self._priority == 'low':
        #     bash_cmd = 'nice -n -5 ' + bash_cmd

        if self._priority == 'mps':
            bash_cmd = 'export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=' + str(self._thread_percentage) + ' && ' + bash_cmd
        if self._job_name == 'megatron-gpt':
            bash_cmd = 'export PYTHONPATH=$PYTHONPATH:/depdency/Megatron-LM' + ' && ' + 'export CUDA_DEVICE_MAX_CONNECTIONS=1' + ' && ' + bash_cmd

        cmd = []
        if self._priority == 'mig-high':
            if self._job_name == 'megatron-gpt':
                # data parallel for multiple MIG slices is not supported yet
                assert(False)
                cmd = [
                    'docker', 'run', '--rm',
                        '--name', self.container_name,
                        '--gpus', '"device=0:0,1:0"',
                        '--ipc', 'host',
                        '--network', 'host',
                        '--cap-add', 'sys_nice',
                        '-u', 'root',
                        '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
                ]
            else:
                cmd = [
                    'docker', 'run', '--rm',
                        '--name', self.container_name,
                        '--gpus', '"device=1:0"',
                        '--ipc', 'host',
                        '--network', 'host',
                        '--cap-add', 'sys_nice',
                        '-u', 'root',
                        '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
                ]
        elif self._priority == 'mig-low':
            if self._job_name == 'megatron-gpt':
                if self._job_id == 2:
                    cmd = [
                        'docker', 'run', '--rm',
                            '--name', self.container_name,
                            '--gpus', '"device=0:1"',
                            '--ipc', 'host',
                            '--network', 'host',
                            '--cap-add', 'sys_nice',
                            '-u', 'root',
                            '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
                    ]
                elif self._job_id == 3:
                    cmd = [
                        'docker', 'run', '--rm',
                            '--name', self.container_name,
                            '--gpus', '"device=1:1"',
                            '--ipc', 'host',
                            '--network', 'host',
                            '--cap-add', 'sys_nice',
                            '-u', 'root',
                            '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
                    ]
            else:
                cmd = [
                    'docker', 'run', '--rm',
                        '--name', self.container_name,
                        '--gpus', '"device=1:1"',
                        '--ipc', 'host',
                        '--network', 'host',
                        '--cap-add', 'sys_nice',
                        '-u', 'root',
                        '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
                ]
        else:
            cmd = [
                'docker', 'run', # '--rm',
                    '--name', self.container_name,
                    '--gpus', f'"device={self._gpus}"',
                    '--ipc', 'host',
                    '--network', 'host',
                    '--cap-add', 'sys_nice',
                    '-u', 'root',
                    '--cpuset-cpus', '0-9,20-29', # FIXME: hard code
            ]
        cmd += self.mounts(mount)

        root_path = os.path.abspath('.')
        if self._antman_config != None:
            cmd += ['-v', root_path + '/' + self._antman_config + ':/gpu_config.json']
        if self._antman_status != None:
            cmd += ['-v', root_path + '/' + self._antman_status + ':/gpu_status.json']

        envs = {
            'TGS_WORKER_IP': str(self._scheduler_ip),
            'TGS_WORKER_PORT': '6889',
            'TGS_TRAINER_PORT': str(self.get_idle_port()),
            'TGS_JOB_ID': str(self._job_id),
            # 'CUDA_VISIBLE_DEVICES' : self._gpus,
            'CUDA_MPS_PIPE_DIRECTORY' : '/tmp/nvidia-mps',
            'GPU_CONFIG_FILE': '/gpu_config.json',
            'GPU_STATUS_FILE': '/gpu_status.json',
        }
        if self.need_throughput == True:
            envs['TGS_LOG_FILE_PATH'] = '/cluster/results/' + self.container_name + '_' + self._job_name + '.txt'
        
        for key, value in envs.items():
            cmd += ['-e', key + '=' + value]

        cmd += [
            self.image_name,
            'bash', '-c', bash_cmd,
        ]

        with open(self.log_path, 'w+') as f:
            self._handler = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                env=envs,
                # shell=True
            )

        return cmd
    

    def terminate(self):
        subprocess.run(['docker', 'kill', self.container_name])
        self._handler.wait()
    

    @property
    def return_code(self):
        return self._handler.poll()


    @property
    def log_path(self):
        if not os.path.exists('job_logs'):
            os.mkdir('job_logs')
        return 'job_logs/' + str(self._job_id) + '_' + str(self._job_name) + '.txt'


    @property
    def container_name(self):
        return f'job_{self._job_id}'


    # @property
    # def image_name(self):
    #     if self._priority in ['high', 'low', 'Ex', 'Co-ex', 'mps']:
    #         return 'tf_torch'
    #     else:
    #         raise Exception()
    

    def update(self, finished_iterations):
        self._finished_iterations += finished_iterations
        throughput = finished_iterations * 10 / (time() - self.last_time)
        self.throughputs.append(throughput)
        self.last_time = time()
        return throughput


    def record(self, timestamp, writer):
        self._timestamp = timestamp
        if len(self.throughputs) > 1:
            writer.save(self)
        self.throughputs = list()
