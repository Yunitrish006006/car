/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * This is an example script for AODV manet routing protocol. 
 *
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

 #include <iostream>
 #include <sstream>
 #include <cmath>
 #include "ns3/aodv-module.h"
 #include "ns3/core-module.h"
 #include "ns3/network-module.h"
 #include "ns3/internet-module.h"
 #include "ns3/mobility-module.h"
 #include "ns3/point-to-point-module.h"
 #include "ns3/v4ping-helper.h"
 #include "ns3/yans-wifi-helper.h"
 
 using namespace ns3;
 
 class AodvExample 
 {
 public:
   AodvExample ();
   bool Configure (int argc, char **argv);
   void Run ();
   void Report (std::ostream & os);
   
 private:
   // 模擬參數
   uint32_t size;
   double step;
   double totalTime;
   bool pcap;
   bool printRoutes;
   
   // 網路容器
   NodeContainer nodes;
   NetDeviceContainer devices;
   Ipv4InterfaceContainer interfaces;
   
 private:
   void CreateNodes ();
   void CreateDevices ();
   void InstallInternetStack ();
   void InstallApplications ();
 };
 
 int main (int argc, char **argv)
 {
   AodvExample test;
   if (!test.Configure (argc, argv))
     NS_FATAL_ERROR ("Configuration failed. Aborted.");
     
   test.Run ();
   test.Report (std::cout);
   return 0;
 }
 
 AodvExample::AodvExample () :
   size (10),           // 節點數量，這裡以 10 個節點作示範
   step (50),           // 節點間距 50 公尺
   totalTime (100),     // 模擬總時長 100 秒
   pcap (true),
   printRoutes (true)
 {
 }
 
 bool
 AodvExample::Configure (int argc, char **argv)
 {
   SeedManager::SetSeed (12345);
   CommandLine cmd (__FILE__);
   cmd.AddValue ("pcap", "Write PCAP traces.", pcap);
   cmd.AddValue ("printRoutes", "Print routing table dumps.", printRoutes);
   cmd.AddValue ("size", "Number of nodes.", size);
   cmd.AddValue ("time", "Simulation time, s.", totalTime);
   cmd.AddValue ("step", "Grid step, m", step);
   cmd.Parse (argc, argv);
   return true;
 }
 
 void
 AodvExample::Run ()
 {
   CreateNodes ();
   CreateDevices ();
   InstallInternetStack ();
   InstallApplications ();
   
   std::cout << "Starting simulation for " << totalTime << " s ...\n";
   Simulator::Stop (Seconds (totalTime));
   Simulator::Run ();
   Simulator::Destroy ();
 }
 
 void
 AodvExample::Report (std::ostream &)
 {
 }
 
 void
 AodvExample::CreateNodes ()
 {
   std::cout << "Creating " << size << " nodes " << step << " m apart.\n";
   nodes.Create (size);
   // 為每個節點命名
   for (uint32_t i = 0; i < size; ++i)
     {
       std::ostringstream os;
       os << "node-" << i;
       Names::Add (os.str (), nodes.Get (i));
     }
   // 設定固定位置佈局（直線排列）
   MobilityHelper mobility;
   mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue (0.0),
                                  "MinY", DoubleValue (0.0),
                                  "DeltaX", DoubleValue (step),
                                  "DeltaY", DoubleValue (0),
                                  "GridWidth", UintegerValue (size),
                                  "LayoutType", StringValue ("RowFirst"));
   mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
   mobility.Install (nodes);
 }
 
 void
 AodvExample::CreateDevices ()
 {
   WifiMacHelper wifiMac;
   wifiMac.SetType ("ns3::AdhocWifiMac");
   YansWifiPhyHelper wifiPhy;
   YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
   wifiPhy.SetChannel (wifiChannel.Create ());
   WifiHelper wifi;
   wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", 
                                 "DataMode", StringValue ("OfdmRate6Mbps"), 
                                 "RtsCtsThreshold", UintegerValue (0));
   devices = wifi.Install (wifiPhy, wifiMac, nodes);
   
   if (pcap)
     {
       wifiPhy.EnablePcapAll ("aodv");
     }
 }
 
 void
 AodvExample::InstallInternetStack ()
 {
   AodvHelper aodv;
   // 安裝網路協議堆疊並設定使用 AODV 作為路由協議
   InternetStackHelper stack;
   stack.SetRoutingHelper (aodv);
   stack.Install (nodes);
   
   Ipv4AddressHelper address;
   address.SetBase ("10.0.0.0", "255.0.0.0");
   interfaces = address.Assign (devices);
   
   if (printRoutes)
     {
       Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("aodv.routes", std::ios::out);
       aodv.PrintRoutingTableAllAt (Seconds (8), routingStream);
     }
 }
 
 void
 AodvExample::InstallApplications ()
 {
   // 使用 V4PingHelper 建立 Ping 應用，由節點 0 對最後一個節點（目標）發送 ICMP 封包
   V4PingHelper ping (interfaces.GetAddress (size - 1));
   ping.SetAttribute ("Verbose", BooleanValue (true));
   
   ApplicationContainer p = ping.Install (nodes.Get (0));
   // 延遲 5 秒開始 Ping，以確保路由建立
   p.Start (Seconds (5));
   p.Stop (Seconds (totalTime) - Seconds (0.001));
   
   // 調整中間節點的移動動作：
   // 將中間節點（node-5，size/2）的移動時間延後到模擬後 1/3（約 66 秒），
   // 並將位置從原有位置移動到 (800, 0, 0)，使其仍能保持部分連線，不會完全斷鏈
   Ptr<Node> node = nodes.Get (size / 2);
   Ptr<MobilityModel> mob = node->GetObject<MobilityModel> ();
   Simulator::Schedule (Seconds (2 * totalTime / 3), &MobilityModel::SetPosition, mob, Vector (800, 0, 0));
 }
 