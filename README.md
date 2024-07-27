# Project Details

To run this script in a distributed setting:

1. Save this script as `distributed_mnist.py`.

2. Set up your cluster configuration. For example, to run on two machines:

   On the first machine (chief):
   ```
   export TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0}}'
   ```

   On the second machine:
   ```
   export TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1}}'
   ```

   Replace `localhost` with the actual IP addresses if running on different machines.

3. Run the script on both machines:
   ```
   python distributed_mnist.py
   ```

Key points about this script:

- It uses `MultiWorkerMirroredStrategy` for distributed training.
- The global batch size is calculated based on the number of workers.
- Checkpoints are saved during training.
- The model is evaluated on the test set after training.
- The final model is saved by the chief worker.

Remember to adjust the `TF_CONFIG` environment variable according to your actual cluster configuration. Also, ensure that all machines have the necessary dependencies installed and can communicate with each other over the network.