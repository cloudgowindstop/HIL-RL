from __future__ import annotations

import threading
import uuid
from typing import Optional
import time
from tqdm import tqdm

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from loguru import logger


class Ros2NodeManager:

    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> Ros2NodeManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self._node: Optional[Node] = None
            self._spin_thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()
            self._paused = False  # Flag to control spinning.
            self._ref_count = 0
            self._ref_count_lock = threading.Lock()
            self._initialized = True
            
            logger.info("ROS 2 Node Manager singleton instance created.")
    
    def initialize(self, node_name: str = "xrocs_shared_ros2_node") -> Node:
        with self._lock:
            if self._node is not None:
                with self._ref_count_lock:
                    self._ref_count += 1
                logger.info(f"ROS 2 node already initialized. Reference count: {self._ref_count}.")
                return self._node
            
            # Initialize ROS 2 context.
            if not rclpy.ok():
                rclpy.init()
                logger.info("ROS 2 context initialized.")
            
            # Create unique node name.
            unique_id = uuid.uuid4().hex[:6]
            full_node_name = f"{node_name}_{unique_id}"
            self._node = Node(full_node_name)
            
            # Start spin thread.
            self._stop_event.clear()
            self._spin_thread = threading.Thread(
                target=self._spin_wrapper,
                daemon=True,
                name=f"ROS2_Shared_Spin_Thread_{unique_id}"
            )
            self._spin_thread.start()
            
            with self._ref_count_lock:
                self._ref_count = 1
            
            logger.success(f"ROS 2 shared node '{full_node_name}' initialized. Reference count: {self._ref_count}.")
            return self._node
    
    def _spin_wrapper(self) -> None:
        executor = SingleThreadedExecutor()
        executor.add_node(self._node)
        
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                if self._paused:
                    time.sleep(0.01)
                    continue
                
                executor.spin_once(timeout_sec=None)

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in ROS 2 shared spin thread.")
        finally:
            executor.remove_node(self._node)
            executor.shutdown()
            logger.info("ROS 2 shared spin thread terminated.")
    
    def acquire(self) -> Node:
        if self._node is None:
            return self.initialize()
        
        with self._ref_count_lock:
            self._ref_count += 1
        
        logger.info(f"Node reference acquired. Reference count: {self._ref_count}.")
        return self._node
    
    def release(self) -> None:
        with self._ref_count_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
            
            logger.info(f"Node reference released. Reference count: {self._ref_count}.")
            
            # Only shutdown if no more references.
            if self._ref_count == 0:
                self._shutdown_internal()
    
    def _shutdown_internal(self) -> None:
        if self._node is None:
            return
        
        logger.info("Shutting down ROS 2 shared node...")
        
        try:
            # Signal stop event.
            self._stop_event.set()
            
            # Wait for spin thread.
            if self._spin_thread and self._spin_thread.is_alive():
                logger.info("Waiting for ROS 2 spin thread to terminate...")
                self._spin_thread.join(timeout=3.0)
            
            # Destroy node.
            try:
                self._node.destroy_node()
                logger.info("ROS2 shared node destroyed.")
            except Exception as e:
                logger.error(f"Error destroying ROS2 node: {e}.")

            self._node = None
            self._spin_thread = None
            
            logger.success("ROS 2 shared node shutdown completed.")
            
        except Exception as e:
            logger.error(f"Error during ROS2 shared node shutdown: {e}.")


# Global singleton instance.
ros2_node_manager = Ros2NodeManager()
for _ in tqdm(range(15), position=0, leave=True, desc="Warming up ROS 2 Node Manager"):
    time.sleep(0.1)
