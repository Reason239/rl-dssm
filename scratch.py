import comet_ml

exp = comet_ml.Experiment(project_name='test')
exp.set_name('Connection test')
for i in range(100):
    exp.log_text('Blah')
