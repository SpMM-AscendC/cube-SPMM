import tbe.common.platform as tbe_platform

def display_soc_specs(soc_version):
    """
    设置 SOC 版本并显示其硬件规格。
    """

    tbe_platform.set_current_compile_soc_info(soc_version)
    print(f"--- {soc_version} 的硬件规格 ---")

    # 要查询的硬件属性列表
    spec_keys = [
        "SOC_VERSION",
        "AICORE_TYPE",
        "CORE_NUM",
        "UB_SIZE",
        "L2_SIZE",
        "L1_SIZE",
        "CUBE_SIZE",
        "L0A_SIZE",
        "L0B_SIZE",
        "L0C_SIZE",
        "SMASK_SIZE"
    ]

    # 遍历列表，获取并打印每个属性的值
    for key in spec_keys:
        value = tbe_platform.get_soc_spec(key)
        print(f"{key}: {value}")
    
    print("-------------------------------------------------")


if __name__ == "__main__":
    target_soc = "Ascend910B2"
    display_soc_specs(target_soc)
