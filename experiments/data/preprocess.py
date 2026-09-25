"""CLI entry point for preprocessing raw datasets into `.ts` files.

Reused from the MuViS benchmark codebase.

Sources
-------
- https://arxiv.org/abs/2603.24602
- https://github.com/noah-puetz/MuViS
"""

import os
import argparse
import logging

from experiments.data.converters import (REVSConverter,
                                         VehicleDynamicsConverter,
                                         TennesseeEastmanProcessConverter,
                                         BeijingPMQualityConverter,
                                         Panasonic18650PFConverter,
                                         PPGDaliaConverterRandomSplit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--raw_folder", type=str, default="data/raw/", help="Input parent folder")
    parser.add_argument("--processed_folder", type=str, default="data/processed/", help="Output parent folder")
    parser.add_argument("--datasets", type=str, default="all", help="Datasets to process, either the name of a dataset or 'all'", choices=["all", "REVS", "VehicleDynamics", "TennesseeEastmanProcess", "PATH", "BeijingPMQuality", "Panasonic18650PF", "PPGDalia"])
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    
    logging_format = "%(asctime)s %(levelname)s:  %(message)s" 
    logging.basicConfig(level=getattr(logging, args.log_level), format=logging_format)


    if args.datasets == "all" or args.datasets == "REVS":
        logging.info("Processing REVS datasets...")
        REVSConverter(
            raw_dir=os.path.join(args.raw_folder, "REVS/2013_Targa_Sixty_Six"), 
            output_dir=os.path.join(args.processed_folder, "REVS/2013_Targa_Sixty_Six"),
            sequence_length=20,
            sequence_step=1,
            ).run()
        REVSConverter(
            raw_dir=os.path.join(args.raw_folder, "REVS/2014_Targa_Sixty_Six"), 
            output_dir=os.path.join(args.processed_folder, "REVS/2014_Targa_Sixty_Six"),
            sequence_length=20,
            sequence_step=1,
            ).run()
        REVSConverter(
            raw_dir=os.path.join(args.raw_folder, "REVS/2013_Monterey_Motorsports_Reunion"), 
            output_dir=os.path.join(args.processed_folder, "REVS/2013_Monterey_Motorsports_Reunion"),
            sequence_length=20,
            sequence_step=1,
            ).run()
        logging.info("Finished processing REVS datasets.")

    if args.datasets == "all" or args.datasets == "VehicleDynamics":
        logging.info("Processing VehicleDynamics datasets...")
        VehicleDynamicsConverter(
            raw_dir=os.path.join(args.raw_folder, "VehicleDynamicsDataset"),
            output_dir=os.path.join(args.processed_folder, "VehicleDynamicsDataset"),
            sequence_length=50,
            sequence_step=1,
            ).run()

    if args.datasets == "all" or args.datasets == "TennesseeEastmanProcess":
        logging.info("Processing TennesseeEastmanProcess dataset...")
        TennesseeEastmanProcessConverter(
            raw_dir=os.path.join(args.raw_folder, "TennesseeEastmanProcess"),
            output_dir=os.path.join(args.processed_folder, "TennesseeEastmanProcess"),
            sequence_length=20,
            sequence_step=1,
            ).run()
        logging.info("Finished processing TennesseeEastmanProcess dataset.")

    if args.datasets == "all" or args.datasets == "BeijingPMQuality":
        logging.info("Processing BeijingPMQuality datasets:")
        
        BeijingPMQualityConverter(
            raw_dir=os.path.join(args.raw_folder, "BeijingPM25Quality"),
            output_dir=os.path.join(args.processed_folder, "BeijingPM25Quality")
            ).clean_data()
        
        BeijingPMQualityConverter(
            raw_dir=os.path.join(args.raw_folder, "BeijingPM10Quality"),
            output_dir=os.path.join(args.processed_folder, "BeijingPM10Quality")
            ).clean_data()
        logging.info("Finished processing BeijingPMQuality datasets.")
        
    if args.datasets == "all" or args.datasets == "Panasonic18650PF":
        logging.info("Processing Panasonic18650PF dataset...")
        Panasonic18650PFConverter(
            raw_dir=os.path.join(args.raw_folder, "Panasonic18650PFData"),
            output_dir=os.path.join(args.processed_folder, "Panasonic18650PFData"),
            sequence_length=120,
            sequence_step=1,
            ).run()
        logging.info("Finished processing Panasonic18650PF dataset.")
        
    if args.datasets == "all" or args.datasets == "PPGDalia":
        PPGDaliaConverterRandomSplit(
            raw_dir=os.path.join(args.raw_folder, "PPGDalia/PPG_FieldStudy"),
            output_dir=os.path.join(args.processed_folder, "PPGDalia"),
            sequence_length=512,
            sequence_step=128
        ).run()
        logging.info("Finished processing PPGDalia dataset.")