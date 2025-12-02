import argparse
from src.data_processing.l2d_load_data import data_downloader
from src.data_processing.l2d_process_pqts import process_tabular_data
from src.data_processing.l2d_process_tags import add_data_tags
from src.data_processing.l2d_process_frames import process_frames
from src.data_processing.l2d_process_lanes import process_lanes_directory
from src.data_processing.l2d_generate_graphs import generate_graphs
from src.data_processing.l2d_annotation_processing import process_annotations_directory

def default_l2d_processing(min_ep, max_ep=-1,
                           run_download=False,
                           run_tabular=False,
                           run_tags=False,
                           run_frames=False,
                           run_lanes=False,
                           run_annotations=False,
                           run_graphs=False):
    
    if not any([run_download, run_tabular, run_tags, run_frames, run_lanes, run_annotations, run_graphs]):
        run_download = run_tabular = run_tags = run_frames = run_lanes = run_annotations = run_graphs = True

    if run_download:
        print("========== Downloading Data ==========")
        data_downloader(min_ep, max_ep, n_secs=3,
             features={"tabular": True,
                     "frames": {
                         'observation.images.front_left': True,
                         'observation.images.left_backward': False,
                         'observation.images.left_forward': False,
                         'observation.images.map': False,
                         'observation.images.rear': False,
                         'observation.images.right_backward': False,
                         'observation.images.right_forward': False,
                     }},
             tabular_data_dir='./data/raw/L2D/tabular',
             frames_dir='./data/raw/L2D/frames',
             metadata_dir='./data/raw/L2D/metadata',
         )

    if run_tabular:
        print("========== Processing Tabular Data ==========")
        process_tabular_data(min_ep, max_ep,
                             overwrite=True, process_columns=True,
                             process_osm=True, process_turning=True,
                             time_sleep=2,
                             source_dir='./data/raw/L2D/tabular',
                             output_dir_processed='./data/processed/L2D',
                             output_dir_tags='./data/semantic_tags/L2D')

    if run_tags:
        print("========== Adding Tags ==========")
        add_data_tags(min_ep, max_ep,
                      data_dir='./data/processed/L2D',
                      tags_dir='./data/semantic_tags/L2D')

    if run_frames:
        if max_ep == -1: verbose = True
        else: verbose = False
        print("========== Processing Frames ==========")
        process_frames(min_ep, max_ep,
                       cameras_on=["observation.images.front_left"],
                       run_dict={"detection": True,
                                 "depth": True,
                                 "speed": True,
                                 'overwrite': True},
                       verbose=verbose,
                       input_base_dir='./data/raw/L2D/frames',
                       output_base_dir='./data/processed_frames/L2D')

    if run_lanes:
        if max_ep == -1: verbose = True
        else: verbose = False
        print("========== Processing Lanes ==========")
        process_lanes_directory(min_ep, max_ep,
                                raw_frames_dir='./data/raw/L2D/frames',
                                yolo_annotations_dir='./data/processed_frames/L2D',
                                output_dir='./data/processed_frames/L2D_lanes',
                                verbose=verbose)

    if run_annotations:
        print("========== Processing Annotations ==========")
        process_annotations_directory(min_ep, max_ep,
            input_dir_original='./data/processed_frames/L2D',
            input_dir_lanes='./data/processed_frames/L2D_lanes',
            output_dir='./data/annotations/L2D',
            original_annotations_folder_name='front_left_Annotations',
            lanes_annotations_folder_name='front_left_Enhanced_LaneAnnotations'
        )

    if run_graphs:
        print("========== Generating Graphs ==========")
        generate_graphs(min_ep, max_ep,
                        source_data_dir='./data/processed/L2D',
                        processed_frame_dir='./data/annotations/L2D',
                        output_dir='./data/graphical/L2D')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process L2D data.")
    parser.add_argument("--min_ep", type=int, default=0, help="Minimum episode number to process.")
    parser.add_argument("--max_ep", type=int, default=-1, help="Maximum episode number to process.")
    parser.add_argument("--download", action="store_true", help="Run data download step.")
    parser.add_argument("--process_tabular", action="store_true", help="Run tabular data processing step.")
    parser.add_argument("--add_tags", action="store_true", help="Run tag processing step.")
    parser.add_argument("--process_frames", action="store_true", help="Run frame processing step.")
    parser.add_argument("--process_lanes", action="store_true", help="Run lane processing step.")
    parser.add_argument("--process_annotations", action="store_true", help="Run annotation processing step.")
    parser.add_argument("--generate_graphs", action="store_true", help="Run graph generation step.")
    parser.add_argument("--all", action="store_true", help="Run all steps (default if no flags are set).")

    args = parser.parse_args()
    
    min_episode = args.min_ep
    max_episode = args.max_ep

    default_l2d_processing(
        min_episode, max_episode,
        run_download=args.download or args.all,
        run_tabular=args.process_tabular or args.all,
        run_tags=args.add_tags or args.all,
        run_frames=args.process_frames or args.all,
        run_lanes=args.process_lanes or args.all,
        run_annotations=args.process_annotations or args.all,
        run_graphs=args.generate_graphs or args.all
    )