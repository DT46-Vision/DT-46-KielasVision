#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import sys

def parse_package_xml(package_xml_path):
    deps = {
        "depend": [],
        "build_depend": [],
        "exec_depend": [],
        "build_export_depend": [],
        "test_depend": []
    }
    try:
        tree = ET.parse(package_xml_path)
        root = tree.getroot()
        for tag in deps.keys():
            for elem in root.findall(tag):
                deps[tag].append(elem.text.strip())
    except Exception as e:
        print(f"解析 {package_xml_path} 出错: {e}")
    return deps

def find_ros2_dependencies(workspace_src):
    results = {}
    for root, dirs, files in os.walk(workspace_src):
        if "package.xml" in files:
            package_path = os.path.join(root, "package.xml")
            try:
                tree = ET.parse(package_path)
                pkg_name = tree.getroot().find("name").text.strip()
            except Exception as e:
                pkg_name = os.path.basename(root)
                print(f"获取包名失败，用目录名代替: {pkg_name}, 错误: {e}")
            
            results[pkg_name] = parse_package_xml(package_path)
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 deps_finder.py <src>")
        sys.exit(1)
    
    src_path = sys.argv[1]
    deps_map = find_ros2_dependencies(src_path)
    
    for pkg, deps in deps_map.items():
        print(f"包: {pkg}")
        for dtype, dlist in deps.items():
            if dlist:
                print(f"  {dtype}:")
                for d in dlist:
                    print(f"    - {d}")
        print("")
