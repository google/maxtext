"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Sweep across inference microbenchmarks."""

import sys
import argparse
import json
import jsonlines
import inference_microbenchmark
import pyconfig


# def main(config, sweep_args):
def main(config):

  # inference_microbenchmark_sweep_key_value_axis_order_product_id_list = [
  #   item for item in config.inference_microbenchmark_sweep_key_value_axis_order_product_id_list.split(':')
  # ]
  inference_microbenchmark_sweep_ar_key_axis_order_list = [
    item for item in config.inference_microbenchmark_sweep_ar_key_axis_order_list.split(':')
  ]
  inference_microbenchmark_sweep_ar_value_axis_order_list = [
    item for item in config.inference_microbenchmark_sweep_ar_value_axis_order_list.split(':')
  ]

  results = []
  for (
    # key_value_axis_order_product_id,
    ar_key_axis_order,
    ar_value_axis_order,
  ) in zip(
    # inference_microbenchmark_sweep_key_value_axis_order_product_id_list,
    inference_microbenchmark_sweep_ar_key_axis_order_list,
    inference_microbenchmark_sweep_ar_value_axis_order_list,
  ):
    # config.key_value_axis_order_product_id = key_value_axis_order_product_id
    config.ar_key_axis_order = ar_key_axis_order
    config.ar_value_axis_order = ar_value_axis_order
    # print(f"@@config.key_value_axis_order_product_id {config.key_value_axis_order_product_id}")
    print(f"@@config.ar_key_axis_order {config.ar_key_axis_order}")
    print(f"@@config.ar_value_axis_order {config.ar_value_axis_order}")
    results = inference_microbenchmark.main(config)
    dimensions_json = {}
    # dimensions_json['key_value_axis_order_product_id'] = key_value_axis_order_product_id
    dimensions_json['ar_key_axis_order'] = ar_key_axis_order
    dimensions_json['ar_value_axis_order'] = ar_value_axis_order
    dimensions_json = {
      **dimensions_json,
      # **json.loads(sweep_args.additional_metadata_metrics_to_save)
      **json.loads(config.inference_microbenchmark_sweep_additional_metadata)
    }
    final = {'metrics': results, 'dimensions': dimensions_json}
    print(f"final {final}")
    results.append(final)
  
  path = 'inference_microbenchmark_sweep_results.jsonl'
  with jsonlines.open(path, mode="w") as writer:
    writer.write_all()


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # # Lists of KV-related settings over which to loop. 
  # # Each iteration will take same index from every list; therefore lists must be of the same length
  # parser.add_argument(
  #     "--key-value-axis-order-product-id-list",
  #     type=str,
  #     default="",
  #     required=True,
  # )
  # parser.add_argument(
  #     "--ar-key-axis-order-list",
  #     type=str,
  #     help='semicolon delimited list of ar key axis orders',
  #     default="1,2,0,3",
  #     required=True,
  # )
  # parser.add_argument(
  #     "--ar-value-axis-order-list",
  #     type=str,
  #     help='semicolon delimited list of ar value axis orders',
  #     default="1,2,0,3",
  #     required=True,
  # )
  # parser.add_argument(
  #     "--save-sweep-result",
  #     action="store_true",
  #     help="Specify to save benchmark results to a jsonlines file",
  # )
  # parser.add_argument(
  #     "--additional-metadata-metrics-to-save",
  #     type=str,
  #     help=(
  #         "Additional metadata about the workload. Should be a dictionary in"
  #         " the form of a string."
  #     ),
  # )
  # sweep_args = parser.parse_args()
  pyconfig.initialize(sys.argv)
  # main(pyconfig.config, sweep_args)
  main(pyconfig.config)