import argparse
from src.data_processing.final_post_processing import (ego_processing_l2d, env_processing_l2d, veh_processing_l2d, ped_processing_l2d,
                                                                       ego_processing_nup, env_processing_nup, veh_processing_nup, ped_processing_nup, 
                                                                       obj_processing_nup)

parser = argparse.ArgumentParser(description="Final Processing of Graphical Data")
parser.add_argument(
    '--process_l2d',
    type=bool,
    default=False,
    help='True/False whether to run L2D final processing'
)
parser.add_argument(
    '--process_nuplan_boston',
    type=bool,
    default=False,
    help='True/False whether to run nuPlan Boston final processing'
)
parser.add_argument(
    '--process_nuplan_pittsburgh',
    type=bool,
    default=False,
    help='True/False whether to run nuPlan Pittsburgh final processing'
)
args = parser.parse_args()

def graph_post_processing(process_l2d=False,process_nuplan_boston=False,process_nuplan_pittsburgh=False):
    
    if process_l2d:
        print('=========== Final Processing for L2D ============')
        in_dir = './data/graphical/L2D/'
        out_dir = './data/graphical_final/L2D'
        annot_dir = "./data/processed_frames/L2D/"

        ego_processing_l2d(input_dir=in_dir, output_dir=out_dir)
        env_processing_l2d(input_dir=out_dir)
        veh_processing_l2d(input_dir=out_dir, annotation_root=annot_dir,hfov_deg=90)
        ped_processing_l2d(out_dir)
    
    if process_nuplan_boston:
        print('============ Final Processing for nuPlan Boston ============')
        in_dir = './data/graphical/nuplan_boston/'
        out_dir = './data/graphical_final/nuplan_boston'

        ego_processing_nup(input_dir=in_dir, output_dir=out_dir)
        env_processing_nup(out_dir)
        veh_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        ped_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        obj_processing_nup(input_dir=out_dir,raw_dir=in_dir)

    if process_nuplan_pittsburgh:
        print('============ Final Processing for nuPlan Pittsburgh ===========')
        in_dir = './data/graphical/nuplan_pitssburgh/'
        out_dir = './data/graphical_final/nuplan_pitssburgh'

        ego_processing_nup(input_dir=in_dir, output_dir=out_dir)
        env_processing_nup(out_dir)
        veh_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        ped_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        obj_processing_nup(input_dir=out_dir,raw_dir=in_dir)

if __name__ == "__main__":
    graph_post_processing(args.process_l2d, args.process_nuplan_boston, args.process_nuplan_pittsburgh)