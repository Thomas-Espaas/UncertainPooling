import UncertainModels

if __name__ == '__main__':
    print('Running')

    for problem in ['Haverly_1', 'Haverly_2', 'Haverly_3', 'Foulds_2']:
        for standard_deviation in [0.2, 0.02, 0.002, 0.0002]:
            for proxy_model in [UncertainModels.ScenarioPooling, UncertainModels.RobustPooling,
                                UncertainModels.StochPooling]:
                uncertain_pooling_object = proxy_model(problem, std=standard_deviation)

                uncertain_pooling_object.solve_problem()

                uncertain_pooling_object.ss_evaluator()

                uncertain_pooling_object.save_results()
