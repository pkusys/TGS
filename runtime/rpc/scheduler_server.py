import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))


from runtime.rpc_stubs.trainer_to_scheduler_pb2 import ReportStatsRequest, ReportStatsResponse, RegisterTrainerRequest, RegisterTrainerResponse
from runtime.rpc_stubs.trainer_to_scheduler_pb2_grpc import TrainerToSchedulerServicer
import runtime.rpc_stubs.trainer_to_scheduler_pb2_grpc as t2s_rpc

import grpc
from concurrent import futures


class SchedulerServerForTrainer(TrainerToSchedulerServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()

        self._logger = logger
        self._callbacks = callbacks
    

    def RegisterTrainer(self, request: RegisterTrainerRequest, context):
        # return super().RegisterTrainer(request, context)
        assert 'RegisterTrainer' in self._callbacks
        register_trainer_impl = self._callbacks['RegisterTrainer']

        success = register_trainer_impl(request.trainer_ip, request.trainer_port, request.job_id)
        response = RegisterTrainerResponse(success=success)

        return response
    

    def ReportStats(self, request: ReportStatsRequest, context) -> ReportStatsResponse:
        # return super().ReportStats(request, context)
        assert 'ReportStats' in self._callbacks
        report_stats_impl = self._callbacks['ReportStats']

        success = report_stats_impl(request.job_id, request.finished_iterations)
        response = ReportStatsResponse(success=success)
        
        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    t2s_rpc.add_TrainerToSchedulerServicer_to_server(SchedulerServerForTrainer(logger, callbacks), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f'worker, rpc, start, server @ {port}')
    
    # server.wait_for_termination()
    return server