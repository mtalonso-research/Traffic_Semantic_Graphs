import argparse
from src.data_processing.final_post_processing import (ego_processing_l2d, env_processing_l2d, veh_processing_l2d, ped_processing_l2d,
                                                                       ego_processing_nup, env_processing_nup, veh_processing_nup, ped_processing_nup, 
                                                                       obj_processing_nup)
from src.data_processing.filtering import filter_json_files, filter_episodes_by_frame_count

def graph_post_processing(process_l2d=False,process_nuplan_boston=False,process_nuplan_pittsburgh=False, process_nuplan_las_vegas=False, min_frames=5):

    if process_l2d:
        print('=========== Final Processing for L2D ============')
        in_dir = './data/graphical/L2D/'
        temp_dir = './data/graphical_final/L2D_temp/'
        out_dir = './data/graphical_final/L2D/'
        annot_dir = "./data/annotations/L2D/"

        filter_episodes_by_frame_count(input_dir=in_dir, output_dir=out_dir, min_frames=min_frames)
        ego_processing_l2d(input_dir=out_dir, output_dir=out_dir)
        env_processing_l2d(input_dir=out_dir)
        veh_processing_l2d(input_dir=out_dir, annotation_root=annot_dir)
        ped_processing_l2d(input_dir=out_dir, annotation_root=annot_dir)
    
    if process_nuplan_boston:
        print('============ Final Processing for nuPlan Boston ============')
        in_dir = './data/graphical/nuplan_boston/'
        temp_dir = './data/graphical/nuplan_boston_temp/'
        out_dir = './data/graphical_final/nuplan_boston'

        filter_json_files(source_dir=in_dir,output_dir=temp_dir)
        filter_episodes_by_frame_count(input_dir=temp_dir, output_dir=out_dir, min_frames=min_frames)
        ego_processing_nup(input_dir=out_dir, output_dir=out_dir)
        env_processing_nup(out_dir)
        veh_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        ped_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        obj_processing_nup(input_dir=out_dir,raw_dir=in_dir)

    if process_nuplan_pittsburgh:
        print('============ Final Processing for nuPlan Pittsburgh ===========')
        in_dir = './data/graphical/nuplan_pittsburgh/'
        temp_dir = './data/graphical/nuplan_pitts_temp/'
        out_dir = './data/graphical_final/nuplan_pittsburgh'

        filter_json_files(source_dir=in_dir,output_dir=temp_dir)
        filter_episodes_by_frame_count(input_dir=temp_dir, output_dir=out_dir, min_frames=min_frames)
        ego_processing_nup(input_dir=out_dir, output_dir=out_dir)
        env_processing_nup(out_dir)
        veh_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        ped_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        obj_processing_nup(input_dir=out_dir,raw_dir=in_dir)

    if process_nuplan_las_vegas:
        print('============ Final Processing for nuPlan Las Vegas ===========')
        in_dir = './data/graphical/nuplan_las_vegas/'
        temp_dir = './data/graphical/nuplan_vegas_temp/'
        out_dir = './data/graphical_final/nuplan_las_vegas'

        filter_json_files(source_dir=in_dir,output_dir=temp_dir)
        filter_episodes_by_frame_count(input_dir=temp_dir, output_dir=out_dir, min_frames=min_frames)
        ego_processing_nup(input_dir=out_dir, output_dir=out_dir)
        env_processing_nup(out_dir)
        veh_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        ped_processing_nup(input_dir=out_dir, raw_dir=in_dir)
        obj_processing_nup(input_dir=out_dir,raw_dir=in_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Processing of Graphical Data")
    parser.add_argument(
        '--process_l2d',
        action='store_true',
        help='Run L2D final processing'
    )
    parser.add_argument(
        '--process_nuplan_boston',
        action='store_true',
        help='Run nuPlan Boston final processing'
    )
    parser.add_argument(
        '--process_nuplan_pittsburgh',
        action='store_true',
        help='Run nuPlan Pittsburgh final processing'
    )
    parser.add_argument(
        '--process_nuplan_las_vegas',
        action='store_true',
        help='Run nuPlan Las Vegas final processing'
    )
    
    args = parser.parse_args()
    graph_post_processing(args.process_l2d, args.process_nuplan_boston, args.process_nuplan_pittsburgh, args.process_nuplan_las_vegas)