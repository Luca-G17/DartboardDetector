import matplotlib.pyplot as plt
import numpy as  np

def Cascade_training_info(stages):
    stages = np.log10(stages.T) + 1
    print(stages[0])
    stage_nums = np.arange(len(stages[0]))
    stage_labels = [ f'Stage {n}' for n in stage_nums]
    print(stage_nums)

    plt.bar(stage_nums - 0.2, stages[0], 0.4, label='True Positive Rate (TPR)')
    plt.bar(stage_nums + 0.2, stages[1], 0.4, label='False Positive Rate (FPR)')

    plt.xticks(stage_nums, stage_labels)
    plt.ylim((-5, 2))
    plt.xlabel('Stages')
    plt.ylabel('Rate: Log10([0-1]) + 1')
    plt.legend()
    plt.grid()
    plt.show()


cascade_stages = np.array([[1, 0.002], [1, 0.00012064], [1, 0.00001755064]])
Cascade_training_info(cascade_stages)