1. `all_data_[grid_num]_[grid_num].pkl`
   2016.2~2016.9，one time interval is 1h

   shape(T, D, W, H),D=41

   0:risk

   1~24:time_period，(one-hot)

   25~31:day_of_week，(one-hot)

   32:holiday，(one-hot)    

   33:temperature

   34:Clear,(one-hot)

   35:Cloudy，(one-hot)

   36:Rain，(one-hot)

   37:Snow，(one-hot)

   38:Mist，(one-hot)

   39:inflow

   40:outflow

2. `bfc_20_10.pkl`

   The  granularity transformation matrix for Hierarchical Constraint.

3. `grid_node_[grid_num]_[grid_num].pkl`
   map graph data to grid data

   shape (W*H,N)

4. `risk_mask_[grid_num]_[grid_num].pkl`

   shape(W,H)

   top risk region mask

5. `[view_type]_adj_[grid_num]_[grid_num].pkl`

   risk/road/poi similarity graph adjacency matrix

   shape (N,N)

6. `trans_[grid_num]_[grid_num].pkl`

   The transformation matrix for Multi-level Embedding Fusion

   

