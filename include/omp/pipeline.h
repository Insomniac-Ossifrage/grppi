/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_PIPELINE_OMP_H
#define GRPPI_PIPELINE_OMP_H

#ifdef GRPPI_OMP

#include <experimental/optional>

#include <boost/lockfree/spsc_queue.hpp>

namespace grppi{

//Last stage
template <typename Stream, typename Stage>
void stages( parallel_execution_omp &p, Stream& st, Stage && s ){

    //Start task
    typename Stream::value_type item;
    std::vector<typename Stream::value_type> elements;
    long current = 0;
    if(p.is_ordered()){
      item = st.pop( );
      while( item.first ) {
        if(current == item.second){
           s( item.first.value() );
           current ++;
        }else{
           elements.push_back(item);
        }
        for(auto it = elements.begin(); it != elements.end(); it++){
           if((*it).second == current) {
              s((*it).first.value());
              elements.erase(it);
              current++;
              break;
           }
        }
       item = st.pop( );
      }
      while(elements.size()>0){
        for(auto it = elements.begin(); it != elements.end(); it++){
          if((*it).second == current) {
            s((*it).first.value());
            elements.erase(it);
            current++;
            break;
          }
        }
      }
    }else{
      item = st.pop( );
      while( item.first ) {
        s( item.first.value() );
        item = st.pop( );
     }
   }
   //End task
}

template <typename Operation, typename Stream,typename... Stages>
void stages( parallel_execution_omp &p, Stream& st, filter_info<parallel_execution_omp, Operation> & se, Stages && ... sgs ) {
  stages(p,st,std::forward<filter_info<parallel_execution_omp, Operation> &&>( se), std::forward<Stages>( sgs )...) ;
}

template <typename Operation, typename Stream,typename... Stages>
 void stages( parallel_execution_omp &p, Stream& st, filter_info<parallel_execution_omp, Operation> && se, Stages && ... sgs ) {
    if(p.ordering){
       mpmc_queue< typename Stream::value_type > q(p.queue_size,p.lockfree);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype.num_threads; th++){
           #pragma omp task shared(q,se,st,nend)
           {
                 typename Stream::value_type item;
                 item = st.pop( ) ;
                 while( item.first ) {
                     if( se.task(item.first.value()) )
                        q.push( item );
                     else{
                        q.push( std::make_pair( typename Stream::value_type::first_type()  ,item.second) );
                     }
                     item = st.pop();
                 }
                 nend++;
                 if(nend == se.exectype.num_threads){
                    q.push( std::make_pair(typename Stream::value_type::first_type(), -1) );
                 }else{
                    st.push(item);
                 }
           }
       }
       mpmc_queue< typename Stream::value_type > qOut(p.queue_size,p.lockfree);
       #pragma omp task shared (qOut,q)
       {
          typename Stream::value_type item;
          std::vector<typename Stream::value_type> elements;
          int current = 0;
          long order = 0;
          item = q.pop( ) ;
          while(1){
             if(!item.first && item.second == -1){
                 break;
             }
             if(item.second == current){
                if(item.first){
                   qOut.push(std::make_pair(item.first,order));
                   order++;
                }
                current++;
             }else{
                elements.push_back(item);
             }
             for(auto it = elements.begin(); it < elements.end(); it++){
                if((*it).second == current){
                    if((*it).first){
                        qOut.push(std::make_pair((*it).first,order));
                        order++;
                    }
                    elements.erase(it);
                    current++;
                    break;
                }
             }
             item=q.pop();
          }
          while(elements.size()>0){
            for(auto it = elements.begin(); it < elements.end(); it++){
              if((*it).second == current){
                  if((*it).first){
                     qOut.push(std::make_pair((*it).first,order));
                     order++;
                  }
                  elements.erase(it);
                  current++;
                  break;
              }
            }
          }
          qOut.push(item);
       }
       stages(p, qOut, std::forward<Stages>(sgs) ... );
       #pragma omp taskwait
      }else{
       mpmc_queue< typename Stream::value_type > q(p.queue_size,p.lockfree);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype.num_threads; th++){
             #pragma omp task shared(q,se,st,nend)
             {
                 typename Stream::value_type item;
                 item = st.pop( ) ;
                 while( item.first ) {
                     if( se.task(item.first.value()) )
                        q.push( item );
//                     else{
//                        q.push( std::make_pair( typename Stream::value_type::first_type()  ,item.second) );
//                     } 
                      item = st.pop();
                 }
                 nend++;
                 if(nend == se.exectype.num_threads){
                    q.push( std::make_pair(typename Stream::value_type::first_type(), -1) );
                 }else{
                    st.push(item);
                 }

          }
       }
       stages(p, q, std::forward<Stages>(sgs) ... );
       #pragma omp taskwait
    }
}


template <typename Operation, typename Stream,typename... Stages>
 void stages( parallel_execution_omp &p, Stream& st, farm_info<parallel_execution_omp, Operation> & se, Stages && ... sgs ) {
 stages(p,st, std::forward< farm_info<parallel_execution_omp, Operation> && >(se), std::forward<Stages>( sgs )...) ;
}

template <typename Operation, typename Stream,typename... Stages>
 void stages( parallel_execution_omp &p, Stream& st, farm_info<parallel_execution_omp, Operation> && se, Stages && ... sgs ) {
  
 
    mpmc_queue< std::pair < std::experimental::optional < typename std::result_of< Operation(typename Stream::value_type::first_type::value_type) >::type >, long > > q(p.queue_size,p.lockfree);
    std::atomic<int> nend ( 0 );
    for( int th = 0; th < se.exectype.num_threads; th++){
      #pragma omp task shared(nend,q,se,st)
      {
         auto item = st.pop();
         while( item.first ) {
         auto out = std::experimental::optional< typename std::result_of< Operation(typename Stream::value_type::first_type::value_type) >::type >( se.task(item.first.value()) );

          q.push( std::make_pair(out,item.second)) ;
          item = st.pop( );
        }
        st.push(item);
        nend++;
        if(nend == se.exectype.num_threads)
          q.push(std::make_pair(std::experimental::optional< typename std::result_of< Operation(typename Stream::value_type::first_type::value_type) >::type >(), -1));
      }              
    }
    stages(p, q, std::forward<Stages>(sgs) ... );
    #pragma omp taskwait
}




//Intermediate stages
template <typename Stage, typename Stream,typename ... Stages>
void stages(parallel_execution_omp &p, Stream& st, Stage && se, Stages && ... sgs ) {

    //Create new queue
    mpmc_queue<std::pair< std::experimental::optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type >, long >> q(p.queue_size,p.lockfree);
    //Start task
    #pragma omp task shared( se, st, q )
    {
        typename Stream::value_type item;
        item = st.pop( ); 
        while( item.first ) {
            auto out = std::experimental::optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type > ( se(item.first.value()) );

            q.push( std::make_pair(out, item.second) );
            item = st.pop(  ) ;
        }
        q.push( std::make_pair(std::experimental::optional< typename std::result_of< Stage(typename Stream::value_type::first_type::value_type) > ::type>(),-1) ) ;
    }
    //End task
    //Create next stage
    stages(p, q, std::forward<Stages>(sgs) ... );
//    #pragma omp taskwait
}

//First stage
template <typename FuncIn, typename = typename std::result_of<FuncIn()>::type,
          typename ...Stages,
          requires_no_arguments<FuncIn> = 0>
void pipeline(parallel_execution_omp &p, FuncIn && in, Stages && ... sts ) {

    //Create first queue
    mpmc_queue<std::pair< typename std::result_of<FuncIn()>::type, long>> q(p.queue_size,p.lockfree);

    //Create stream generator stage
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task shared(in,q)
            {
                long order = 0;
                while( 1 ) {
                    auto k = in();
                    q.push( std::make_pair(k,order) ) ;
                    order++;
                    if( !k ) 
                        break;
                }
            }
            //Create next stage
            stages(p, q, std::forward<Stages>(sts) ... );
//            stages(p, q, sts ... );
            #pragma omp taskwait
        }
    }
}

}
#endif

#endif