import { Component, OnInit } from '@angular/core';
import * as myGlobals from '../globals';
import { Product } from '../modals/Product';
import { ProductsService } from '../services/products.service';
@Component({
  selector: 'app-basket-page',
  templateUrl: './basket-page.component.html',
  styleUrls: ['./basket-page.component.scss']
})
export class BasketPageComponent implements OnInit{
  test:string=""
  products:Product[]=[];
  quantities:number[]=[];
  totalPrice:string[]=[];
  constructor(public productsService:ProductsService){}

  getBasket()
  {
    this.products=[]
    this.quantities=[]
    this.totalPrice=[]
    for(let product of myGlobals.basket)
    {
      let productModal:Product=this.productsService.getProduct(product.id);
      this.products.push(productModal)
      this.quantities.push(product.quantity);
      this.totalPrice.push((product.quantity*parseFloat(productModal.price)).toFixed(2))
    }
  }

  ngOnInit(){
    this.getBasket();
  }

  removeFromBasket(id:number){
    this.productsService.removeFromBasket(id);
    this.getBasket();
  }

}
