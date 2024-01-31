process predict {
    label 'withGPU'

    container { params.segmentation_container }
    cpus { params.segmentation_cpus }
    memory { params.segmentation_memory }

    input:
    val(image_path)
    val(ch)
    val(scale)
    val(model_path)
    val(output_path)

    output:
    tuple val(image_path), val(output_path)
    
    script:
    """
    /entrypoint.sh segmentation -i $image_path -o $output_path -n $ch/$scale
    """
}