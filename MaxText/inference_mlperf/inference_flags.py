import jax
import os

prefill_1024_xla_flags = {
    "xla_jf_auto_cross_replica_sharding": "False",
    "xla_tpu_decompose_all_gather_einsum": "True",
    "xla_tpu_enable_windowed_einsum_for_reduce_scatter": "True",
    "xla_tpu_enable_async_collective_fusion": "False",
    "xla_tpu_enable_async_collective_fusion_fuse_all_gather": "False",
    "xla_tpu_overlap_compute_collective_tc": "False",
    "xla_all_gather_latency_bound_threshold_in_bytes": "524290",
    # Enable more RS fusions.
    "xla_tpu_relayout_group_size_threshold_for_reduce_scatter": 1,
    # Tuned flags
    "xla_jf_rematerialization_percent_shared_memory_limit": "98",
    "xla_tpu_allocate_scoped_vmem_at_same_offset": "False",
    "xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers": ("NO_SCALE"),
    "xla_tpu_async_copy_bandwidth_scaling_factor": "1.0044",
    "xla_tpu_copy_elision_analysis_allowance": "239621",
    "xla_tpu_copy_fusion_pad_unpad_ratio": "242.6129",
    "xla_tpu_copy_insertion_use_region_analysis_limit": "3441",
    "xla_tpu_dot_dot_fusion": "True",
    "xla_tpu_dot_dot_fusion_duplicated": "False",
    "xla_tpu_enable_aggressive_broadcast_priority_update": "True",
    "xla_tpu_enable_dot_strength_reduction": "True",
    "xla_tpu_enable_experimental_fusion_cost_model": "True",
    "xla_tpu_enable_vmem_to_vmem_dmas": "False",
    "xla_tpu_enforce_prefetch_fifo_order": "True",
    "xla_tpu_layout_use_dot_grouping": "False",
    "xla_tpu_memory_bound_loop_optimizer_options": "enabled:true",
    "xla_tpu_msa_inefficient_use_to_copy_ratio": "0.471",
    "xla_tpu_nd_short_transfer_max_chunks": "4415",
    "xla_tpu_order_dot_after_layout": "False",
    "xla_tpu_perform_spmd_cse_prevention": "True",
    "xla_tpu_prefetch_interval_picker_size_override": "26672104",
    "xla_tpu_reduce_loop_fusion_dup_with_unfusable_user": "False",
    "xla_tpu_rwb_fusion": "False",
    "xla_tpu_scavenge_vmem_for_fusions": "False",
    "xla_tpu_scoped_vmem_limit_kib": "19592",
    "xla_tpu_sliced_prefetch_max_slices": "0",
    "xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions": "True",
    "xla_tpu_use_repeated_instance_for_preferred_prefetch_time": "False",
    "xla_tpu_vector_load_fusion_window": "644",
    "xla_tpu_vector_store_fusion_window": "1228",
    "xla_vf_vmem_enable_cross_program_prefetch_freeing": "False",
    "xla_vf_vmem_max_outstanding_evictions": "136",
    "xla_vf_vmem_max_outstanding_prefetches": "131",
    "xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio": "16.0009",
    "xla_vf_vmem_max_repacks": "18",
    "xla_vf_vmem_max_retries": "2",
    "xla_vf_vmem_min_overlap_to_async_copy_ratio": "1.4973",
    "xla_vf_vmem_preferred_overlap_to_async_copy_ratio": "8.3221",
}


def get_xla_flags(xla_flags_dict):
    return ' '.join(f"--{k}={v}" for k, v in xla_flags_dict.items())
    