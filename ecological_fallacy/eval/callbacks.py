import datetime
from pathlib import Path
import pandas as pd
import transformers


class EvalResultsCallback(transformers.TrainerCallback):
    """
    A [`TrainerCallback`] that logs evaluation results to a CSV file
    """

    def __init__(self, experiment_path: str, experiment_config):
        self.experiment_path = Path(experiment_path)
        self.experiment_config = experiment_config
        self.num_logged = 0

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            with open(f'{self.experiment_path}/result.csv', 'a') as f:
                eval_logs = state.log_history[self.num_logged:]
                eval_logs = [
                    log for log in eval_logs
                    #only keep log entries that contain eval results
                    if any(["eval_" in str(k) for k in log.keys()])
                ]
                for log in eval_logs:
                    log['timestamp'] = str(datetime.datetime.now())
                    log.update(self.experiment_config)
                self.num_logged = len(state.log_history)
                df = pd.DataFrame(eval_logs)
                df = df.set_index('timestamp')
                df.to_csv(f, header=not f.tell())