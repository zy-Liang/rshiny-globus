from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.engines import HighThroughputEngine
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface

user_opts = {
    'armis2': {
        'worker_init': 'deactivate; module purge; module load gcc cuda/11.7.1 cudnn/11.7-v8.7.0 python3.9-anaconda; source /nfs/turbo/umms-dinov2/LLaMA/1.0.1/bin/activate',
        'scheduler_options': '#SBATCH --gpus=1 #SBATCH --cpus-per-task=1 #SBATCH --mem-per-cpu=100g',
    }
}

config = Config(
    executors=[
        HighThroughputEngine(
            worker_port_range = (8888,8987),
            interchange_port_range = (8888,8987),
            max_workers_per_node=2,
            worker_debug=True,
            address=address_by_interface('eth0'),  # specific for armis2
            provider=SlurmProvider(
                partition='gpu',  # update for slurm -p --partition value
                launcher=SrunLauncher(),
				account='dinov0',  # update for slurm -A --account value
                exclusive=False,

                # string to prepend to #SBATCH blocks in the submit
                # script to the scheduler eg: '#SBATCH --constraint=knl,quad,cache'
                scheduler_options=user_opts['armis2']['scheduler_options'],

                # Command to be run before starting a worker, such as:
                # 'module load Anaconda; source activate parsl_env'.
                worker_init=user_opts['armis2']['worker_init'],

                # Scale between 0-1 blocks with 2 nodes per block
                nodes_per_block=1,
                mem_per_node=100,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,

                # Hold blocks for 30 minutes
                walltime='00:30:00'
            ),
        )
    ],
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "armis2-build-slurm",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}
