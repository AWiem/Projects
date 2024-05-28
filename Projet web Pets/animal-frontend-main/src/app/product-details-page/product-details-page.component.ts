import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Product } from '../modals/Product';
import { ProductsService } from '../services/products.service';

@Component({
  selector: 'app-product-details-page',
  templateUrl: './product-details-page.component.html',
  styleUrls: ['./product-details-page.component.scss']
})
export class ProductDetailsPageComponent implements OnInit{
  product!:Product;
  quantity=1;
  constructor(private route:ActivatedRoute,public productsService:ProductsService){}
  ngOnInit(){
    let id:number=parseInt(this.route.snapshot.params['id'],10);
    this.product=this.productsService.getProduct(id);
  }

}
